
# knowledge-distill

This code implements the knowledge distillation idea presented in the research paper "https://arxiv.org/abs/1503.02531". The code in this GitHub repository, trains a larger network with more parameters which is the teacher model. Then using knowledge distillation technique, a student model with less parameters is trained. The student model reaches reasonably close in accuracy to the teacher model. The implementation downloads and uses the tiny imagenet dataset.


## How to run the code

```sh
python3 tiny_imagenet_distill.py
```

## Results
Training Teacher Model on Tiny ImageNet...
Teacher Epoch [1/10], Loss: 3.7101
Teacher Epoch [2/10], Loss: 2.2122
Teacher Epoch [3/10], Loss: 1.7991
Teacher Epoch [4/10], Loss: 1.5816
Teacher Epoch [5/10], Loss: 1.4353
Teacher Epoch [6/10], Loss: 1.3171
Teacher Epoch [7/10], Loss: 1.2243
Teacher Epoch [8/10], Loss: 1.1440
Teacher Epoch [9/10], Loss: 1.0681
Teacher Epoch [10/10], Loss: 1.0065

Training Student Model via Knowledge Distillation on Tiny ImageNet...
Student Epoch [1/10], Loss: 5.5640
Student Epoch [2/10], Loss: 3.0807
Student Epoch [3/10], Loss: 2.1460
Student Epoch [4/10], Loss: 1.7165
Student Epoch [5/10], Loss: 1.4777
Student Epoch [6/10], Loss: 1.3330
Student Epoch [7/10], Loss: 1.2431
Student Epoch [8/10], Loss: 1.1714
Student Epoch [9/10], Loss: 1.1137
Student Epoch [10/10], Loss: 1.0737

Evaluating Teacher Model...
Teacher Model Accuracy: 65.39%

Evaluating Student Model...
Student Model Accuracy: 63.22%
Total parameters for the teacher model: 11279112
Trainable parameters for the teacher: 11279112
Total parameters for the student model: 2480072
Trainable parameters for the student: 2480072







