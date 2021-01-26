'''

This file contains the final Multimodal Siamese Network which takes in two images and two audio files and predc=icts whether they are from the same person or not.

architechtures.py which have been imported contains the individual face and audio detection models.

'''

import architectures

# siamese network model which helps in one-shot learning

def siamese_network(input_dim_img,input_dim_aud):
    
    img_a = Input(shape=input_dim_img)
    img_b = Input(shape=input_dim_img)
    aud_a = Input(shape=input_dim_aud)
    aud_b = Input(shape=input_dim_aud)

    # extracting the image embeddings from the image network for two input images
    
    img_network = architectures.image_feat_network(input_dim_img)
    
    # load pretrained weights for image network
    img_network.load_weights(' set path ')

    img_network = Model(inputs = img_network.input , outputs = img_network.layers[-2].output)
    
    feat_img_a = img_network(img_a)
    feat_img_b = img_network(img_b)
    
    # extracting the audio embeddings from the image network for two output images

    aud_network = architectures.audio_feat_network(input_dim_aud)
    
    #load pretrained weights for audio network
    aud_network.load_weights(' set path ')

    aud_network = Model(inputs = aud_network.input , outputs = aud_network.layers[-2].output)

    feat_aud_a = aud_network(aud_a)
    feat_aud_b = aud_network(aud_b)

    # concatenate the first image's embeddings with the first audio's embeddings
    concat_a = Concatenate(axis=1)([feat_img_a, feat_aud_a])

    # concatenate the second image's embeddings with the second audio's embeddings
    concat_b = Concatenate(axis=1)([feat_img_b, feat_aud_b])

    input_dim = concat_a.shape[1] # (?) put the input dimensions here

    base_network = architectures.multimodal_network(input_dim)
    
    # left vector encodings
    encoded_l = base_network(concat_a) 
    # right vector encodings
    encoded_r = base_network(concat_b)

    # Layer to merge two encoded inputs with the l1 distance between them
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))

    # Get the L1 Distance between the two encoded legs and predict a sigmoid using that.
    L1_distance = L1_layer([encoded_l, encoded_r])
    prediction = Dense(1,activation='sigmoid')(L1_distance)

    model = Model(inputs=[img_a, aud_a, img_b,aud_b], outputs=prediction)

    optimizer = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999)
    model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['Precision'])

    return optimizer , model