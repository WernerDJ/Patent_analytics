from django import forms
from datetime import datetime
current_year = datetime.now().year

class ExcelUploadForm(forms.Form):
    excel_file = forms.FileField()
    ipc_groups = forms.CharField(
        required=False, 
        label="IPC Groups",
        help_text="Enter one or more IPC groups, separated by commas."
    )
    start_year = forms.IntegerField(
        initial=current_year-20, 
        min_value=current_year-20, 
        max_value=current_year,
        label="Start Year"
    )
    end_year = forms.IntegerField(
        initial=current_year, 
        min_value=current_year -20, 
        max_value=current_year,
        label="End Year"
    )

class ReducedExcelUploadForm(forms.Form):
    excel_file = forms.FileField(label="Upload Excel File")
    start_year = forms.IntegerField(
        initial=current_year-20, 
        min_value=current_year-20, 
        max_value=current_year,
        label="Start Year"
    )
    end_year = forms.IntegerField(
        initial=current_year, 
        min_value=current_year -20, 
        max_value=current_year,
        label="End Year"
    )

class SimpleExcelUploadForm(forms.Form):
    excel_file = forms.FileField(label="Upload Excel File")

