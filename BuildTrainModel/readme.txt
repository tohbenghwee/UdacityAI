Thank for reviewing.
This project consist of the following files.

Part 1 of project :
        ## The jupyter notebook file which used to develop my model
	1) Classifier Project.ipynb 
	## A sample output of after running of the ipynb file
	2) Classifier Project.html

Part 2 of project :
	## For the input argument
	1) input_args.py
	## The base class for Train and Predict
	2) classifier_base.py
	## The required train file for training model
	3) train.py
	## The required predict file for predict model
	4) train.py

	Training command (example) :
	python train.py --device=gpu --arch=vgg16 --epochs=5 --learn_rate=0.001 --hidden_units=4096 --checkpoint=checkpoint.pth	

	Predict command (example) :
	python predict.py --device=gpu --checkpoint=checkpoint.pth --image_path='flowers/valid/10/image_07102.jpg' --topk=5 --class_cat_path=cat_to_name.json