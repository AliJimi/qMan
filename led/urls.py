from django.urls import path

from led.views import led_view, sag_view, numbers_json

app_name = "led"

urlpatterns = [
    path('show/', led_view),
    path('sag/', sag_view, name='sag'),
    path('numbers/', numbers_json, name='numbers'),
]