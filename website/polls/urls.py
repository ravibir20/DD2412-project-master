from django.urls import path

from . import views

urlpatterns = [
    # ex: /polls/
    path('', views.index, name='index'),
    path('answer_questions', views.AnswerAPI.as_view(), name='answer_questions'),
    path('compare_models', views.ModelAPI.as_view(), name='compare_models'),
    path('clear_cookies', views.clear_cookies, name='clear_cookies'),
    path('results', views.results, name='results'),
    path('results_comparison', views.results_comparison, name='results_comparison'),
]