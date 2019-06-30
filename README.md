Fusion Strategy for Prosodic and Lexical Representations of Word Importance
================

This project demonstrates a early feature fusion strategiy for prosodic and lexical representations for predicting importance of words in a spoken dialgoue for understanding its meaning. The model operates on human-annotated corpus of word importance for its training and evaluation. The corpus can be downloaded from: http://latlab.ist.rit.edu/lrec2018


You can cite the methods and/or results of this work using:

> Sushant Kafle, Cecilia O. Alm, and Matt Huenerfauth. 2019. Fusion Strategy for Prosodic and Lexical Representations of Word Importance. In Proceedings of International Speech Communication Association (Interspeech). 2019.


I. Project Structure:
=========

- "data": Folder contains model vocabulary and pre-computed features indexable by word ids.
- "examples": Folder contains a sample of train/eval data used in this model. Each subfolder inside this folder represents a 'conversation'. For example, in the case of Switchboard corpus, this could be a folder for a phone-conversation between two speakers. Inside each 'conversation' folder, there are multiple subfolders representing a dialouge level split of the conversation, which is indexed as "{speaker}-{line_id}", such that "A-51" represents `line` 51 spoken by `speaker` A. Inside this folder there are: 
	* "praat.textgrid" which represents Praat Textgrid contains the timestamp information for the words; 
	* "sentence.wav" is the spoken audio
	* "timestamp.csv" is the timestamp information in the csv format
	* "words" folder which contains all the speech features (pre-extracted using our feature extraction software).

- "saved_models": Folder contains our pre-trained model on the train-split of the word importance corpus.
- "utils": Folder home to all the utility functions.
- "config.py": Model configurations.
- "SpeechTextModel.py": Our word importance prediction model.
- "run.py": Script put together to run the pre-trained word importance model on the example dataset in the "example" folder.


II. Running Examples:
=========

Use `run.py` script to see the output of the word on an example dataset in the "example" folder:

	python run.py

This should give out the output in the form:

	WORD IMPORTANCE PREDICTION OUTPUT
	=================================
	--> um-hum (0.131412) yeah (0.033996) probably (0.125088) the (0.012908) hardest (0.597960) thing (0.189134) in (0.160486) in (0.160380) my (0.070749) family (0.247930) uh (0.001937) my (0.065381) grandmother (0.843788) she (0.131237) had (0.275710) to (0.056653) be (0.195990) put (0.346472) in (0.118089) nursing (0.879508) home (0.563176) and (0.195925) um (0.023632) she (0.130859) had (0.205812) used (0.299910) a (0.009633) walker (0.768561) for (0.167595) for (0.100835) quite (0.217472) sometime (0.374890) probably (0.242587) about (0.170903) six (0.829853) to (0.027557) nine (0.896921) months (0.657620) and (0.187088) um (0.015875)


