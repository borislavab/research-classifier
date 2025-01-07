from django.urls import path
from . import views

urlpatterns = [
    path("predict/", views.predict, name="predict"),
    path("prediction/<str:task_id>/", views.get_prediction, name="get_prediction"),
]
