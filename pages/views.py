import os
from django.conf import settings
from django.shortcuts import render
from django.views.generic import TemplateView, FormView
from django.views import View
from .forms import ExcelUploadForm, SimpleExcelUploadForm, ReducedExcelUploadForm
from django.contrib.auth.mixins import LoginRequiredMixin
# Django cache
from django.core.cache import cache
from django.http import JsonResponse
from django.core.files.storage import default_storage
# Celery Cache 
from celery.result import AsyncResult
#Subir al contenedor S3
import boto3
from botocore.exceptions import ClientError
#Analytic Sections for non logged in users
from analpat.tasks import (analyze_countries_task, 
                            analyze_wordclouds_task, 
                            analyze_IPC_task,
                            analyze_Applicants_task,
                            analyze_ApplicInventNetwork_task
                            )
#Temporary treatment of graphs
from pages.analytic_functions import Patent_Analysis
import pickle
#Graphs generation
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
# Generate unique ID for each class instance created
# So as to avoid two users retrieving, by mistake the others user graphs
from uuid import uuid4  
import sys #just for debugging


class HomePageView(TemplateView):
    template_name = "home.html"

class AboutPageView(TemplateView):
    template_name = "about.html"

# The analytics themselves are carried out, for the non logged in users, in the celery module anaplat/tasks.py, in the background.
# The html templates regularly check for a result to be deliverable, and when it is so, present it.

class CeleryFormMixin:
    task_function = None  # Each subclass must set this

    def form_valid(self, form):
        excel_file = form.cleaned_data.get('excel_file')
        ipc_input = form.cleaned_data.get('ipc_groups', "")
        ipc_list = [ipc.strip() for ipc in ipc_input.split(",") if ipc.strip()]
        start_year = form.cleaned_data.get('start_year')
        end_year = form.cleaned_data.get('end_year')
        time_range = [start_year, end_year]

        safe_name = f"uploads/{uuid4()}_{excel_file.name}"
        stored_rel_path = default_storage.save(safe_name, excel_file)
        file_path = os.path.join(settings.MEDIA_ROOT, stored_rel_path)

        # Dispatch the task (specific to the subclass)
        task = self.task_function.delay(file_path, ipc_list, start_year, end_year)
        return JsonResponse({'task_id': task.id})

class TaskStatusView(View):
    def get(self, request, *args, **kwargs):
        task_id = request.GET.get('task_id')
        if not task_id:
            return JsonResponse({'status': 'error', 'message': 'No task_id provided'})

        task = AsyncResult(task_id)
        if task.state == 'PENDING':
            return JsonResponse({'status': 'pending'})
        if task.state == 'FAILURE':
            return JsonResponse({'status': 'failure', 'error': str(task.result)})
        if task.state == 'SUCCESS':
            return JsonResponse({'status': 'success', 'result': task.result})
        return JsonResponse({'status': task.state})

##################################################################################################################
##                                                  Non logged in analytics                                     ##
##################################################################################################################

class CountriesView(CeleryFormMixin,  FormView):
    template_name = "pages/countries.html"
    form_class = ExcelUploadForm
    task_function = analyze_countries_task

class WordCloudsView(CeleryFormMixin,  FormView):
    template_name = "pages/wordclouds.html"
    form_class = ExcelUploadForm
    task_function = analyze_wordclouds_task

class IPCView(CeleryFormMixin,  FormView):
    template_name = "pages/IPC.html"
    form_class = ReducedExcelUploadForm
    task_function = analyze_IPC_task

class ApplicantsView(CeleryFormMixin,  FormView):
    template_name = "pages/Applicants.html"
    form_class = ReducedExcelUploadForm
    task_function = analyze_Applicants_task

class ApplicInventNetworkView(CeleryFormMixin,  FormView):
    template_name = "pages/network.html"
    form_class = SimpleExcelUploadForm
    task_function = analyze_ApplicInventNetwork_task

##################################################################################################################
##                                                  Logged in analytics                                         ##
##################################################################################################################          
class AllAnalyticsView(LoginRequiredMixin, FormView):
    template_name = "pages/all_analytics.html"
    form_class = ExcelUploadForm

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['has_data'] = self._get_cached_data() is not None
        return context

    def form_valid(self, form):
        excel_file = form.cleaned_data.get('excel_file')
        ipc_input = form.cleaned_data.get('ipc_groups', "")
        ipc_list = [ipc.strip() for ipc in ipc_input.split(",") if ipc.strip()]
        self.start_year = form.cleaned_data.get('start_year')
        self.end_year = form.cleaned_data.get('end_year')
        time_range = [self.start_year, self.end_year]

        try:
            analyzer = Patent_Analysis(excel_file)
            analyzer.filter_by_ipc_and_year(ipc_list, time_range)
            analyzer.start_year = self.start_year
            analyzer.end_year = self.end_year
            
            # Cache the analyzer object
            self._cache_data(analyzer)
            
            extra_context = {
                'upload_success': True,
                'ipc_list': ipc_list,
                'time_range': time_range,
                'has_data': True
            }
            
            context = self.get_context_data(form=form, **extra_context)
            return self.render_to_response(context)
            
        except Exception as e:
            return self.form_invalid(form, exception=e)

    def form_invalid(self, form, exception=None):
        extra_context = {}
        if exception:
            extra_context['error'] = str(exception)
        
        context = self.get_context_data(form=form, **extra_context)
        return self.render_to_response(context)

    def _cache_data(self, analyzer):
        """Cache the analyzer object with Redis"""
        cache_key = f"user_{self.request.user.id}_analyzer"
        cache.set(
            cache_key,
            pickle.dumps(analyzer),
            timeout=3600  # 1 hour expiration
        )

    def _get_cached_data(self):
        """Retrieve cached analyzer object"""
        cache_key = f"user_{self.request.user.id}_analyzer"
        cached = cache.get(cache_key)
        return pickle.loads(cached) if cached else None

    def post(self, request, *args, **kwargs):
        """Handle different analysis types"""
        if 'run_analysis' in request.POST:
            analysis_type = request.POST.get('analysis_type')
            return self._run_analysis(analysis_type)
        return super().post(request, *args, **kwargs)

    def _run_analysis(self, analysis_type):
        """Execute specific analysis using cached data"""
        analyzer = self._get_cached_data()
        if not analyzer:
            extra_context = {'error': "No data available. Please upload a file first."}
            context = self.get_context_data(**extra_context)
            return self.render_to_response(context)

        try:
            if analysis_type == 'wordcloud':
                return self._generate_wordclouds(analyzer)
            elif analysis_type == 'countries':
                return self._generate_country_analysis(analyzer)
            elif analysis_type == 'IPC':
                return self._generate_ipc_analysis(analyzer)
            elif analysis_type == 'applicants':
                return self._generate_applicants_analysis(analyzer)
            elif analysis_type == 'network_analysis':
                return self._generate_network_analysis(analyzer)
            # Add other analysis types here...
            
        except Exception as e:
            extra_context = {'error': f"Analysis failed: {str(e)}"}
            context = self.get_context_data(**extra_context)
            return self.render_to_response(context)

    def _upload_plot(self, plot_obj, prefix):
        """Helper to upload matplotlib Figure, pyplot module, or other image objects to Cloudinary"""
        import matplotlib.figure
        import matplotlib.pyplot as plt
        from PIL import Image
        import types

        filename = f"{prefix}_{uuid4().hex[:8]}.png" #It was uuid.uuid4() before I changed above import uuid by from uuid import uuid4
        temp_path = f"/tmp/{filename}"
        s3_key = f"plots/{filename}"

        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_S3_REGION_NAME
        )

        try:
            if isinstance(plot_obj, matplotlib.figure.Figure):
                # Matplotlib Figure
                plot_obj.savefig(temp_path)
                plt.close(plot_obj)

            elif isinstance(plot_obj, types.ModuleType) and plot_obj is plt:
                # Passed the pyplot module directly
                plt.savefig(temp_path)
                plt.close()

            elif hasattr(plot_obj, "to_file"):  # WordCloud object
                plot_obj.to_file(temp_path)

            elif isinstance(plot_obj, Image.Image):  # PIL Image
                plot_obj.save(temp_path)

            elif hasattr(plot_obj, "save"):  # Other save()-able object
                plot_obj.save(temp_path)

            else:
                raise TypeError(f"Unsupported plot object type: {type(plot_obj)}")

            # Upload to S3
            s3_client.upload_file(
                temp_path,
                settings.AWS_STORAGE_BUCKET_NAME,
                s3_key,
                ExtraArgs={
                    'ContentType': 'image/png'
                    # No ACL needed - bucket policy handles public access
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

    ##################################################################################################################
    ##                                                 The analytics themselves                                     ##
    ################################################################################################################## 

    def _generate_country_analysis(self, analyzer):
        """Generate country-specific analyses"""
        # Prepare priority data
        priority_df = analyzer.prepare_priority_data()
        
        # Generate different plots (not duplicates!)
        plt_priority_years = analyzer.plot_priority_years_bar(priority_df)
        plt_priority_countries = analyzer.plot_priority_countries_bar(priority_df)  # Priority countries
        plt_top_countries = analyzer.plot_top_10_countries()  # Publication countries
        plt_patent_flow = analyzer.analyze_patent_flow(top_n=10)  # Patent flow

        try:
            # Upload to Cloudinary with descriptive names
            priority_years_url = self._upload_plot(plt_priority_years, "priority_years")
            priority_countries_url = self._upload_plot(plt_priority_countries, "priority_countries")  
            top_countries_url = self._upload_plot(plt_top_countries, "top_pub_countries")  # Different from priority
            patent_flow_url = self._upload_plot(plt_patent_flow, "patent_flow")
        except Exception as e:
            print("UPLOAD ERROR:", e)
            raise

        extra_context = {
            'analysis1': "<p>Country analysis complete. See the graphs below.</p>",
            'priority_years_img_url': priority_years_url,
            'priority_countries_img_url': priority_countries_url,  # Priority countries
            'countries_img_url': top_countries_url,  # Publication countries  
            'origin_destcountr_img_url': patent_flow_url,  # Patent flow
            'active_tab': 'countries',
            'has_data': True
        }
        
        context = self.get_context_data(**extra_context)
        return self.render_to_response(context)

    def _generate_wordclouds(self, analyzer):
        """Generate all wordclouds"""
        wcld_nouns = analyzer.generate_wordclouds_by_pos(pospeech='Nouns')
        wcld_verbs = analyzer.generate_wordclouds_by_pos(pospeech='Verbs')
        wcld_adjectives = analyzer.generate_wordclouds_by_pos(pospeech='Adjectives')


        # Upload to Cloudinary and get URLs
        nouns_url = self._upload_plot(wcld_nouns, "wcld_nouns")
        verbs_url = self._upload_plot(wcld_verbs, "wcld_verbs")
        adjectives_url = self._upload_plot(wcld_adjectives, "wcld_adjectives")

        extra_context = {
            'analysis1': "<p>Wordcloud analysis complete. See the graphs below.</p>",
            'wcld_nouns_img': nouns_url,
            'wcld_verbs_img': verbs_url,
            'wcld_adjectives_img': adjectives_url,
            'active_tab': 'wordcloud',
            'has_data': True
        }
        
        context = self.get_context_data(**extra_context)
        return self.render_to_response(context)

    def _generate_ipc_analysis(self, analyzer):
        """Generate IPC-specific analyses"""
        sys.stderr.write(">>> _start_ipc_analysis\n") #debug
        plt_top_ipcs = analyzer.plot_top_ipcs()
        sys.stderr.write(">>> calculate top IPCs\n") #debug
        try:
            ipc_defs_html = analyzer.get_top_ipcs_AI_defined()
            sys.stderr.write(">>> successfully got IPC definitions HTML\n")
        except Exception as e:
            import traceback
            traceback.print_exc()
            ipc_defs_html = f"<p>IPC definitions could not be generated: {e}</p>"
                
        # For parallel coordinates, we need year range from cached data
        # You might need to store this in cache or pass it differently
        plt_parallel = analyzer.plot_parallel_coordinates(top_n=5, year_range=range(analyzer.start_year, analyzer.end_year))

        # Upload to Cloudinary or S3
        ipcs_url = self._upload_plot(plt_top_ipcs, "top_ipcs")
        parallel_url = self._upload_plot(plt_parallel, "parallel_coords")

        extra_context = {
            'analysis1': "<p>IPC analysis complete. See the graphs below.</p>",
            'top_ipcs_img': ipcs_url,
            'top_ipcs_defs_html': ipc_defs_html,
            'parallel_img': parallel_url,
            'active_tab': 'IPC',
            'has_data': True
        }
        
        context = self.get_context_data(**extra_context)
        return self.render_to_response(context)

    def _generate_applicants_analysis(self, analyzer):
        """Generate applicant-specific analyses"""
        plt_top_applicants = analyzer.get_top_non_inventor_applicants(top_n=20)
        plt_applicant_ipc = analyzer.plot_applicant_ipc_bubble_chart(top_n=20)
        plt_applicant_parallel = analyzer.plot_applicant_parallel_coordinates(top_n=5, year_range=(analyzer.start_year, analyzer.end_year))
        sys.stderr.write(">>> _start_applicants_analysis(\n") #debug
        # Upload to Cloudinary
        applicants_url = self._upload_plot(plt_top_applicants, "top_applicants")
        app_ipc_url = self._upload_plot(plt_applicant_ipc, "applicant_ipc_bubble")
        app_parallel_url = self._upload_plot(plt_applicant_parallel, "applicant_parallel")

        # Get top applicant names (already available from analyzer)
        applicants = [name for name, _ in analyzer.top_applicants[:10]]
        sys.stderr.write(">>> _Get top applicant names (already available from analyzer)\n") #debug
        # Ensure top IPCs are counted first
        if not hasattr(analyzer, 'top_ipc_counts'):
            analyzer.plot_top_ipcs()  
        try:
            ai_report = analyzer.get_top_applicants_AI_description(applicants)
            ai_report_html = analyzer.ai_markdown_to_html_table(ai_report)
            sys.stderr.write(">>> _AI-generated report \n")
        except Exception as e:
            import traceback
            traceback.print_exc()
            ai_report = "<p>AI report could not be generated: {}</p>".format(e)

        extra_context = {
            'analysis1': "<p>Applicants analysis complete. See the graphs below.</p>",
            'topAppl_img': applicants_url,
            'Appl_IPC_img': app_ipc_url,
            'ParalleltopApp_img': app_parallel_url,
            'ai_report': ai_report_html,
            'active_tab': 'applicants',
            'has_data': True
        }
        import pprint
        pprint.pprint(extra_context)
        context = self.get_context_data(**extra_context)
        return self.render_to_response(context)

    def _generate_network_analysis(self, analyzer):    
        try:
            sys.stderr.write(">>> Entered _generate_network_analysis\n") #debug
            sys.stderr.flush() #debug
            
            # Now all Patent_Analysis instances have network methods
            # step 1: refine applicant/inventor data
            analyzer.filter_data_for_network()  
            # step 2: build the graph
            analyzer.build_network_graph()
            sys.stderr.write(f"Graph has {analyzer.network_graph.number_of_nodes()} nodes\n") #debug
            sys.stderr.flush() #debug
            # step 3: generate the network image
            plt_network = analyzer.generate_network_image(top_n=20)
            sys.stderr.write(">>> Finished generating figure\n") #debug
            sys.stderr.flush() #debug
            network_url = self._upload_plot(plt_network, "plt_network")
            connections_df = analyzer.get_connections_df()
            connections_df_html = analyzer.connections_to_html_table(connections_df)

            extra_context = {
                'network_img': network_url,
                'active_tab': 'network_analysis',
                'connections_html':connections_df_html,
                'has_data': True,
                'analysis1': "<p>Network analysis complete. See the graph below.</p>",
            }

            context = self.get_context_data(**extra_context)
            return self.render_to_response(context)
        except Exception as e:
            import traceback
            print("❌ ERROR in _generate_network_analysis:", e)
            traceback.print_exc()
            return None
