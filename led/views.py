from django.http import JsonResponse
from django.shortcuts import render

import random


def led_view(request):
    return render(request, 'led.html')


def sag_view(request):
    return render(request, 'sag.html')


def numbers_json(request):
    return JsonResponse([
        random.randint(1, 100),
        random.randint(1, 100),
        random.randint(1, 100)
    ], safe=False)
