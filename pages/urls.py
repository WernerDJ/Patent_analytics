from django.urls import path
from .views import (HomePageView, 
                    AboutPageView, 
                    WordCloudsView, 
                    IPCView, 
                    ApplicantsView, 
                    ApplicInventNetworkView, 
                    CountriesView, 
                    TaskStatusView, 
                    AllAnalyticsView # , filter_data
                    )
urlpatterns = [
    path("", HomePageView.as_view(), name="home"),
    path('task-status/', TaskStatusView.as_view(), name='task_status'),
    path("countries", CountriesView.as_view(), name="countries"),
    path("wordclouds/", WordCloudsView.as_view(), name="wordclouds"),
    path("IPC/", IPCView.as_view(), name="IPC"),
    path("Applicants/", ApplicantsView.as_view(), name="Applicants"),
    path("network/", ApplicInventNetworkView.as_view(), name = "network"),
    path("All Graphs", AllAnalyticsView.as_view(), name ="everything"),
    path("about/", AboutPageView.as_view(), name="about"),
]
