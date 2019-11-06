The whole experiment mainly has three sub mdoels:
(1) basic image caption model
(2) transformer caption model
(3) face recognition model
The transformer caption is able to be joint trained with face recognition model.


Image caption part:
    Here are the manual of basic caption model and transformer caption model:
    Step1: Prepare training data
        (1) Download the COCO dataset and coco captions
            for COCO captions, please download it from (http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)
                Extract dataset_coco.json from the zip file and copy it in to data/. This file provides preprocessed captions and also standard train-val-test splits.
            for COCO dataset, please download it from (http://mscoco.org/dataset/#download)
                We need 2014 training images and 2014 val. images. You should put the train2014/ and val2014/ in the same directory, denoted as $IMAGE_ROOT
        (2) Prepare labels:
            $ ./prepare_image_labels.sh
            prepro_labels.py will map all words that occur <= 5 times to a special UNK token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into data/cocotalk.json and discretized caption data are dumped into data/cocotalk_label.h5.
        (3) Do pre-cnn feature extraction
            $ ./prepare_image_features.sh
            prepro_feats.py extract the resnet50 features (both fc feature and last conv feature) of each image. The features are saved in data/cocotalk_fc and data/cocotalk_att, and resulting files are about 50GB.

    Step2: Prepare project relayment
        (1) clone the coco-caption project, which url is (https://github.com/tylin/coco-caption)

    Step3: Train the basic baseline model
        (1) adjust the parameters on 'train.baseline.sh' according to your own condition
            TOOLPATH1 should be directed to cooc-caption project.
            TOOLPATH2 should be directed to cider folder, cider is already put into this project, so you don't have to download it again.

        (2) you should change the python interaptor to your own python2 interaptor, strongly recommand to install anaconda2, which contains mostly package you need to train this model.

        (3) then make sure about that the cocotalk.json, cocotalk_fc, cocotalk_att, cocotalk_label.h5 are all on the right place as I set.

        (4) CUDA_VISIBLE_DEVICES=0 means you run this code on the first GPU on your pc, you can try two gpus by setting CUDA_VISIBLE_DEVICES=0, 1

        (5) after all the settings complete, you should run:
            $ ./train.baseline.sh
            after training, the checkpoints will be saved on the "log_fc" folder.

    Step4: Train the transformer model:
        (1) set the parameters as you did on Step3, this scripts has a new paramets you should be notice,  the "despective_model_on" should be set to 0 when you just want't to train the transformer model.
        (2) after setting, you should run:
            $ ./train.transformer.sh

    Step5: Evaluation
        (1) after you trained a baseline model or a transformer model, you can run a evaluation.
        (2) you should check the eval.sh scripts, change the image_folder to the path where you put the under eval images.
        (3) num_images equals -1 means you test all image on the folder, otherwise the num means how many images will be test on the folder.
        (4) to visiualize some eval results:
            $ cd vis
            $ python -m SimpleHTTPServer
            the visit host is: localhost:8000, if you do this on a server, you can try: server_ip:8000
        (5) you can evaluate model on karpathy's test split by trying:
            $ ./eval_on_karpathy.sh

    Attention:
        (1) The whole training scripts only works on machine with cuda.
        (2) you should use python2 because some packages don't have python3 version.
        (3) if all settings are set correctly, it should look like that:
        ➜  image-caption ./train.baseline.sh
            DataLoader loading json file:  data/cocotalk.json
            vocab size is  9487
            DataLoader loading h5 file:  data/cocotalk_fc data/cocotalk_att data/cocotalk_box data/cocotalk_label.h5
            max sequence length in data is 16
            read 123287 image features
            assigned 113287 images to split train
            assigned 5000 images to split val
            assigned 5000 images to split test
            Read data: 0.281188964844
            iter 0 (epoch 0), train_loss = 9.157, time/batch = 0.146
            Read data: 0.0435318946838
            iter 1 (epoch 0), train_loss = 8.655, time/batch = 0.088
            Read data: 0.033087015152
            iter 2 (epoch 0), train_loss = 8.001, time/batch = 0.072
            Read data: 0.0275020599365
            iter 3 (epoch 0), train_loss = 7.806, time/batch = 0.065
            Read data: 0.030375957489
            iter 4 (epoch 0), train_loss = 7.391, time/batch = 0.083
            Read data: 0.0395522117615
            iter 5 (epoch 0), train_loss = 7.140, time/batch = 0.073
            Read data: 0.0314450263977



Face recognition part:


Joint trainig transformer and face model:
    (1) set despective_model_on to 1
    (2) run:
        $ ./train.transformer.sh
