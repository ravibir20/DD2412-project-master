from django.http import HttpResponse
from .models import Image, Answer, Comparison, Prediction,MODEL_CHOICES, EXPLANATION_CHOICES
from django.views import View
from django.shortcuts import render
import json
import random
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

def index(request):
    return render(request, 'index.html')

def results_comparison(request):
    data_one = []
    for modelA, _ in MODEL_CHOICES:
        for modelB,_ in [model for model in MODEL_CHOICES if model[0] != modelA]:
            for explanation,_ in EXPLANATION_CHOICES:
                configuration = modelA+"_"+modelB+"_"+explanation
                sum = 0
                nTests = 0
                for comparison in Comparison.objects.filter(explanation = explanation):
                    if comparison.modelA == modelA and comparison.modelB == modelB:
                        sum += comparison.answer
                        nTests += 1
                    elif comparison.modelB == modelA and comparison.modelA == modelB:
                        sum -= comparison.answer
                        nTests += 1
                if nTests == 0:
                    avg = None
                else:
                    avg = sum/nTests
                data_one.append({
                    "config": configuration,
                    "average": avg
                })
    data_model = {
        "vgg":0,
        "resnet":0,
        "n_vgg":0,
        "n_resnet":0,
    }
    for comparison in Comparison.objects.all():
        data_model[comparison.modelA] += comparison.answer
        data_model[comparison.modelB] -= comparison.answer
        data_model["n_"+comparison.modelA] += 1
        data_model["n_"+comparison.modelB] += 1
    if data_model["n_vgg"] != 0:
        data_model["vgg_percent"] = data_model["vgg"]/data_model["n_vgg"]
    else:
        data_model["vgg_percent"] = 0

    if data_model["n_resnet"] != 0:
        data_model["resnet_percent"] = data_model["resnet"]/data_model["n_resnet"]
    else:
        data_model["resnet_percent"] = 0
    
    return render(request, 'compare_models.html', {"data_one":data_one, "data_model":data_model})
    


def results(request):
    images = list(Image.objects.all())
    question_results = []
    total = {}
    for model,_ in MODEL_CHOICES:
        for explanation,_ in EXPLANATION_CHOICES:
            config = model+"_"+explanation
            total[config+"_correct"] = 0
            total[config+"_total"] = 0
    
    for image in images:
        image_result = {}
        image_result["image"] = image.ground_truth
        image_result["label"] = image.true_label
        for model, _ in MODEL_CHOICES:
            for explanation,_ in EXPLANATION_CHOICES:
                config = model+"_"+explanation
                answers = Answer.objects.filter(image=image, model=model, explanation=explanation)
                correct_answers = 0
                if len(answers) != 0:
                    for answer in answers:
                        if answer.answer == image.true_label:
                            correct_answers += 1
                    percent = 100*correct_answers/len(answers)
                else:
                    percent = 0
                total[config+"_correct"] += correct_answers
                total[config+"_total"] += len(answers)
                image_result[config+"_n"] = correct_answers
                image_result[config+"_percent"] = percent
        question_results.append(image_result)
    for model,_ in MODEL_CHOICES:
        for explanation,_ in EXPLANATION_CHOICES:
            config = model+"_"+explanation
            if total[config+"_total"] != 0:
                total[config+"_percent"] = 100*total[config+"_correct"]/total[config+"_total"]
            else:
                total[config+"_percent"] = 0
    return render(request, 'results.html', {"results":question_results, "total": total})


@method_decorator(csrf_exempt, name='dispatch')
class AnswerAPI(View):       
    def get(self, request):
        cookie = request.COOKIES.get('answered_images')
        if cookie:
            answered_images = json.loads(cookie)
        else:
            answered_images = []
        if len(answered_images) >= 30:
            response = redirect('compare_models')
            return response
            #return render(request, 'all_done.html')
        
        # Get all options
        images = list(Image.objects.exclude(id__in = [obj[0] for obj in answered_images]))
        models = MODEL_CHOICES
        explanations = EXPLANATION_CHOICES

        # Find a non-answered question
        found_a_good_question = False

        # Search for a good question
        nSearches = 0
        while not found_a_good_question and nSearches < 20:        
            image = random.choice(images)
            model = random.choice(models)[0]
            explanation = random.choice(explanations)[0]

            if [str(image.id), model, explanation] not in answered_images:
                source = explanation+"_"+model+"_image"
                try:
                    target_image = getattr(image, source)
                    if target_image:
                        found_a_good_question = True
                except:
                    pass
            nSearches += 1
        
        if found_a_good_question:
            response = render(request, 'one_question.html', {'image':image,'target_image': target_image, "model": model, "explanation": explanation, "random":random.uniform(0,1)})
        else:
            response = render(request, 'all_done.html')
        
        return response
        
    def post(self, request):
        # Get data
        cookie = request.COOKIES.get('answered_images')
        if cookie:
            answered_images = json.loads(cookie)
        else:
            answered_images = []
        data = request.body.decode("utf-8").split("&")
        query = { a.split("=")[0] : a.split("=")[1] for a in data }
        response = redirect('answer_questions')
        try:
            query["answer"] = query["answer"].replace("+", " ")

            image = Image.objects.get(id=query["image_id"])
            answer = list(Prediction.objects.filter(prediction=query["answer"]))[0]
            #Save data
            answer = Answer.objects.create(image = image, model = query["model"], explanation = query["explanation"], answer = answer)
        
            #answer.save()
            answered_images.append([query["image_id"] ,query["model"], query["explanation"]])
            response.set_cookie("answered_images", value=json.dumps(answered_images))
        except:
            pass
        return response


@method_decorator(csrf_exempt, name='dispatch')
class ModelAPI(View):       
    def get(self, request):
        cookie = request.COOKIES.get('answered_comparisons')
        if cookie:
            answered_comparisons = json.loads(cookie)
        else:
            answered_comparisons = []
        if len(answered_comparisons) >= 15:
            return render(request, 'all_done.html')
        # Get all options
        images = list(Image.objects.all())
        models = MODEL_CHOICES
        explanations = EXPLANATION_CHOICES

        # Find a non-answered question
        found_a_good_question = False

        # Search for a good question
        nSearches = 0
        while not found_a_good_question and nSearches < 20:        
            image = random.choice(images)
            
            modelA = random.choice(models)
            modelB = random.choice([model for model in models if model is not modelA])[0]
            modelA = modelA[0]

            explanation = random.choice(explanations)[0]

            if [str(image.id), modelA, modelB, explanation] not in answered_comparisons:
                sourceA = explanation+"_"+modelA+"_image"
                sourceB = explanation+"_"+modelB+"_image"
                predSourceA = modelA+"_prediction"
                predSourceB = modelB+"_prediction"

                try:
                    target_imageA = getattr(image, sourceA)
                    target_imageB = getattr(image, sourceB)
                    predict_imageA = getattr(image, predSourceA)
                    predict_imageB = getattr(image, predSourceB)
                    if target_imageA and target_imageB and predict_imageA==predict_imageB:
                        found_a_good_question = True
                except:
                    pass
            nSearches += 1
        
        if found_a_good_question:
            response = render(request, 'one_comparison.html', {'image':image,'target_imageA': target_imageA,'target_imageB': target_imageB, "modelA": modelA, "modelB": modelB, "explanation": explanation, 'prediction': predict_imageA})
        else:
            response = render(request, 'all_done.html')
        
        return response
        
    def post(self, request):
        # Get data
        cookie = request.COOKIES.get('answered_comparisons')
        if cookie:
            answered_comparisons = json.loads(cookie)
        else:
            answered_comparisons = []
        data = request.body.decode("utf-8").split("&")
        query = { a.split("=")[0] : a.split("=")[1] for a in data }

        image = Image.objects.get(id=query["image_id"])
        response = redirect('compare_models')
        try:
            #Save data
            answer = Comparison.objects.create(image = image, modelA=query["modelA"], modelB=query["modelB"], explanation = query["explanation"], answer=int(query["answer"]))
            #answer.save()
            answered_comparisons.append([query["image_id"] ,query["modelA"],query["modelB"], query["explanation"]])
            response.set_cookie("answered_comparisons", value=json.dumps(answered_comparisons))
        except:
            pass
        return response



def clear_cookies(request):
    response = render(request, 'clear_cookies.html')
    response.delete_cookie(key='answered_images')
    response.delete_cookie(key='answered_models')
    return response

# Leave the rest of the views (detail, results, vote) unchanged
def detail(request, question_id):
    return HttpResponse("You're looking at question %s." % question_id)


def vote(request, question_id):
    return HttpResponse("You're voting on question %s." % question_id)
