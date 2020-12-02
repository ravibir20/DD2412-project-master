from django.db import models

MODEL_CHOICES = (
    ('vgg','vgg'),
    ('resnet', 'resnet')
)

EXPLANATION_CHOICES = (
    ('backprop','backprop'),
    ('gradcam', 'gradcam'),
    ('squadcam','squadcam'),
)


class Prediction(models.Model):
    prediction = models.CharField(max_length=200, unique=True)
    def __str__(self):
        return self.prediction


class Image(models.Model):
    image_name = models.CharField(max_length=200)
    ground_truth = models.ImageField(upload_to="static/images/ground_truth", blank=True, null=True)
    
    backprop_vgg_image = models.ImageField(upload_to="static/images/vgg/backprop", blank=True, null=True)
    gradcam_vgg_image = models.ImageField(upload_to="static/images/vgg/gradcam", blank=True, null=True)
    squadcam_vgg_image = models.ImageField(upload_to="static/images/vgg/squadcam", blank=True, null=True)
    
    backprop_resnet_image = models.ImageField(upload_to="static/images/resnet/backprop", blank=True, null=True)
    gradcam_resnet_image = models.ImageField(upload_to="static/images/resnet/gradcam", blank=True, null=True)
    squadcam_resnet_image = models.ImageField(upload_to="static/images/resnet/squadcam", blank=True, null=True)

    choice_one = models.ForeignKey(Prediction, related_name="choice_one",on_delete=models.SET_NULL, null=True, blank=True)
    choice_two = models.ForeignKey(Prediction, related_name="choice_two",on_delete=models.SET_NULL, null=True, blank=True)

    true_label = models.ForeignKey(Prediction, related_name="true_label",on_delete=models.SET_NULL, null=True, blank=True)
    vgg_prediction = models.ForeignKey(Prediction, related_name="vgg_prediction",on_delete=models.SET_NULL, null=True, blank=True)
    resnet_prediction = models.ForeignKey(Prediction, related_name="resnet_prediction",on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return self.image_name + " -> " + self.choice_one.prediction + " " + self.choice_two.prediction

class Answer(models.Model):
    image = models.ForeignKey(Image, on_delete=models.CASCADE, null=True)
    model = models.CharField(max_length=200, choices=MODEL_CHOICES, null=True)
    explanation = models.CharField(max_length=200, choices=EXPLANATION_CHOICES, null=True)
    answer = models.ForeignKey(Prediction, on_delete=models.SET_NULL, null=True, blank=True)
    def __str__(self):
        return self.image.image_name + " answer: "+ str(self.answer)

class Comparison(models.Model):
    image = models.ForeignKey(Image, on_delete=models.CASCADE)
    modelA = models.CharField(max_length=200, choices=MODEL_CHOICES)
    modelB = models.CharField(max_length=200, choices=MODEL_CHOICES)
    explanation = models.CharField(max_length=200, choices=EXPLANATION_CHOICES)
    
    answer = models.IntegerField()
    def __str__(self):
        return self.image.image_name + " answer: "+ str(self.answer)
