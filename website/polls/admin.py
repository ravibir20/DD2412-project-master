from django.contrib import admin

from .models import Image, Answer, Comparison, Prediction

admin.site.register(Image)
admin.site.register(Answer)
admin.site.register(Comparison)
admin.site.register(Prediction)