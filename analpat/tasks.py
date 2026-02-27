from celery import shared_task
from pages.analytic_functions import Patent_Analysis
import os
from uuid import uuid4
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

import boto3
from botocore.exceptions import ClientError
from django.conf import settings

# ✅ Try to set a CJK font if available
try:
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'AR PL UMing CN']:
        if font in available_fonts:
            matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            break
except:
    pass  # Fall back to default fonts



# boto3 will automatically use AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
# from the environment variables passed by docker-compose
s3_client = boto3.client(
    's3',
    region_name=settings.AWS_S3_REGION_NAME
)

def _upload_plot(plot_obj, prefix: str) -> str:
    """
    Save a matplotlib Figure / PIL Image / WordCloud to a temp PNG,
    upload to Cloudinary, return secure_url.
    """
    import matplotlib.figure
    import matplotlib.pyplot as plt
    from PIL import Image
    import types
    filename = f"{prefix}_{uuid4().hex[:8]}.png"    #It was uuid.uuid4() before I changed above import uuid by from uuid import uuid4
    temp_path = f"/tmp/{filename}"
    s3_key = f"plots/{filename}"
    try:
        if isinstance(plot_obj, matplotlib.figure.Figure):
            plot_obj.savefig(temp_path)
            plt.close(plot_obj)
        elif isinstance(plot_obj, Image.Image):
            plot_obj.save(temp_path)
        elif hasattr(plot_obj, "to_file"):
            plot_obj.to_file(temp_path)
        elif hasattr(plot_obj, "save"):
            plot_obj.save(temp_path)
        else:
            # Try pyplot as a last resort
            try:
                plt.savefig(temp_path)
                plt.close()
            except Exception as e:
                raise TypeError(f"Unsupported plot object type: {type(plot_obj)}") from e
        # Upload to S3
        s3_client.upload_file(
            temp_path,
            settings.AWS_STORAGE_BUCKET_NAME,
            s3_key,
            ExtraArgs={
                'ContentType': 'image/png',
            }
        )
        
        # Return public URL
        url = f"https://{settings.AWS_S3_CUSTOM_DOMAIN}/{s3_key}"
        return url
        
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        raise
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@shared_task(bind=True)
def analyze_countries_task(self, excel_file_path, ipc_list=None, start_year=None, end_year=None):
    """
    Async country analysis.
    Args come from the view; excel_file_path must be readable by the worker.
    """
    try:
        ipc_list = ipc_list or []
        # Normalize years (form values may be strings or None)
        sy = int(start_year) if start_year not in (None, "",) else None
        ey = int(end_year)   if end_year   not in (None, "",) else None
        time_range = [sy, ey] if (sy and ey) else None

        analyzer = Patent_Analysis(excel_file_path)
        analyzer.filter_by_ipc_and_year(ipc_list, time_range)

        priority_df = analyzer.prepare_priority_data()

        fig_years = analyzer.plot_priority_years_bar(priority_df)
        fig_priority_countries = analyzer.plot_priority_countries_bar(priority_df)
        fig_pub_countries = analyzer.plot_top_10_countries()
        fig_flow = analyzer.analyze_patent_flow(top_n=10)

        result = {
            "priority_years_img_url": _upload_plot(fig_years, "priority_years"),
            "priority_countries_img_url": _upload_plot(fig_priority_countries, "priority_countries"),
            "countries_img_url": _upload_plot(fig_pub_countries, "top_pub_countries"),
            "origin_destcountr_img_url": _upload_plot(fig_flow, "patent_flow"),
            "message": "Country analysis complete.",
        }

        # Optional cleanup of the uploaded Excel file if it's temporary.
        # Be careful: only do this if you stored it in a temp location.
        # try:
        #     os.remove(excel_file_path)
        # except OSError:
        #     pass

        return result

    except Exception as e:
        raise self.retry(exc=e, countdown=60, max_retries=3)

@shared_task(bind=True)
def analyze_wordclouds_task(self, excel_file_path, ipc_list=None, start_year=None, end_year=None):
    try:
        ipc_list = ipc_list or []
        sy = int(start_year) if start_year not in (None, "",) else None
        ey = int(end_year)   if end_year   not in (None, "",) else None
        time_range = [sy, ey] if (sy and ey) else None

        analyzer = Patent_Analysis(excel_file_path)
        analyzer.filter_by_ipc_and_year(ipc_list, time_range)

        # Generate separate wordclouds
        fig_nouns = analyzer.generate_wordclouds_by_pos(pospeech='Nouns')
        fig_verbs = analyzer.generate_wordclouds_by_pos(pospeech='Verbs')
        fig_adjectives = analyzer.generate_wordclouds_by_pos(pospeech='Adjectives')

        result = {
            "wcld_nouns_img": _upload_plot(fig_nouns, "wcld_nouns"),
            "wcld_verbs_img": _upload_plot(fig_verbs, "wcld_verbs"),
            "wcld_adjectives_img": _upload_plot(fig_adjectives, "wcld_adj"),
            "message": "Wordcloud analysis complete.",
        }

        return result

    except Exception as e:
        raise self.retry(exc=e, countdown=60, max_retries=3)

@shared_task(bind=True)
def analyze_IPC_task(self, excel_file_path, ipc_list=None, start_year=None, end_year=None):
    import matplotlib.pyplot as plt
    
    try:
        sy = int(start_year) if start_year not in (None, "",) else None
        ey = int(end_year)   if end_year   not in (None, "",) else None
        time_range = [sy, ey] if (sy and ey) else None
        
        # Create an instance of the Patent Analysis tools
        analyzer = Patent_Analysis(excel_file_path)
        analyzer.filter_by_ipc_and_year(ipc_list, time_range)
        
        # Generate the graphs ONE AT A TIME, uploading immediately
        # ✅ Generate and upload first plot
        plt_ipcs = analyzer.plot_top_ipcs()
        top_ipcs_url = _upload_plot(plt_ipcs, "plt_ipcs")
        plt.close('all')  # Clean up matplotlib state
        
        # ✅ Generate and upload second plot  
        plt_defs = analyzer.get_top_ipcs_with_titles()
        top_ipcs_defs_url = _upload_plot(plt_defs, "plt_defs")
        plt.close('all')  # Clean up matplotlib state
        
        # ✅ Generate and upload third plot
        plt_parallel = analyzer.plot_parallel_coordinates(top_n=5, year_range=range(sy, ey + 1))
        parallel_url = _upload_plot(plt_parallel, "plt_parallel")
        plt.close('all')  # Clean up matplotlib state
        
        # Save the graphs for the html template
        result = {
            'top_ipcs_img': top_ipcs_url,
            'top_ipcs_defs_img': top_ipcs_defs_url,
            'parallel_img': parallel_url,
            "message": "IPC analysis complete.",
        }
        return result
    except Exception as e:
        raise self.retry(exc=e, countdown=60, max_retries=3)


@shared_task(bind=True)
def analyze_Applicants_task(self, excel_file_path, ipc_list=None, start_year=None, end_year=None):
    try:
        sy = int(start_year) if start_year not in (None, "",) else None
        ey = int(end_year)   if end_year   not in (None, "",) else None
        time_range = [sy, ey] if (sy and ey) else None
        #
        #Create an instance of the Patent Analysis tools
        analyzer = Patent_Analysis(excel_file_path)
        analyzer.filter_by_ipc_and_year(ipc_list, time_range)
        #Generate the graphs 
        plt_topAppl = analyzer.get_top_non_inventor_applicants(top_n=20)
        plt_Appl_IPC = analyzer.plot_applicant_ipc_bubble_chart(top_n=20)
        plt_ParalleltopAppl = analyzer.plot_applicant_parallel_coordinates(top_n=5, year_range=(start_year, end_year + 1))
        #Save the graphs for the html template
        result = {
            'topAppl_img': _upload_plot(plt_topAppl, "plt_topAppl"),
            'Appl_IPC_img': _upload_plot(plt_Appl_IPC , "plt_Appl_IPC "),
            'ParalleltopApp_img': _upload_plot(plt_ParalleltopAppl, "plt_ParalleltopAppl"),
            "message": "Applicants analysis complete.",
            }
        return result
    except Exception as e:
        raise self.retry(exc=e, countdown=60, max_retries=3)        

@shared_task(bind=True)
def analyze_ApplicInventNetwork_task(self, excel_file_path, ipc_list=None, start_year=None, end_year=None):
    try:
        sy = int(start_year) if start_year not in (None, "",) else None
        ey = int(end_year)   if end_year   not in (None, "",) else None
        time_range = [sy, ey] if (sy and ey) else None
        
        # Create an instance of Patent_Analysis (no need for Patent_Network anymore)
        analyzer = Patent_Analysis(excel_file_path)
        

        # ✅ DEBUG: Print some sample applicant names to check encoding
        print("Sample applicant names from raw data:")
        sample_applicants = analyzer.data['Applicants'].dropna().head(20)
        for i, name in enumerate(sample_applicants):
            print(f"  {i+1}. {repr(name)}")  # repr() shows the actual bytes/encoding
        
        # Apply any filtering if needed
        if ipc_list or time_range:
            analyzer.filter_by_ipc_and_year(ipc_list or [], time_range)
        
        # Use the network analysis methods from the base class
        analyzer.filter_data_for_network()
        analyzer.build_network_graph()
        
        # Generate the network graph
        plt_network = analyzer.generate_network_image(top_n=20)
        
        # Save the graphs for the html template
        result = {
            'network_img_url':_upload_plot(plt_network, "plt_network"),  # Fixed the function name
            "message": "Analysis complete. See the graphs below",
        }
        return result
    except Exception as e:
        import traceback
        print(f"Error in network analysis: {traceback.format_exc()}")
        raise self.retry(exc=e, countdown=60, max_retries=3)
