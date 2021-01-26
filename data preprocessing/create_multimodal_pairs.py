def create_same_class_pairs(X,y,check,img_a,img_b,img_y,same_class_list,num):

  i = 0
  #num : number of pairs to be created
  while(i<num):
    
    # perform pairing for same classes.
    if check == 0 :
      same_class = np.random.randint(N_CLASSES)
      same_class_list.append(same_class)

    if check == 1 :
      same_class= same_class_list[i]
    
    enum_list=[k for k,j in enumerate(y) if j==same_class]
    
    if len(enum_list) >= 2:
      index = random.sample( enum_list, 2) 
    x_random = X[index]
    y_same = same_class

    img_a.append(x_random[0]) # these are indexes in X_train
    img_b.append(x_random[1])

    #append the class number for same class pair
    img_y.append(y_same) # this is the class

    i = i+1

  return img_a,img_b,img_y

#function to create pairs of data from different class

def create_diff_class_pairs(X,y,img_a,img_b,img_y,num):

  i = 0
  #num : number of pairs to be created
  while(i<num):
    # Repeat this as many times as many examples you want using a loop

    # select random indices from the shape[0] of the X_train array
    index = np.random.choice(X.shape[0], 2, replace=False)  

    # make a random array by choosing subsets from X_train
    x_random = X[index]
    y_random = y[index]

    # append the first image to img_a and second to img_b

    # finally based on the similarity of the images based on their classes append either the class (if they are similar) or -1 if they are dissimilar
    if(y_random[0] == y_random[1]):
      pass
    else:
      img_a.append(x_random[0])
      img_b.append(x_random[1])

      #append -1 for different class pair
      img_y.append(-1)

      i=i+1
    
  return img_a,img_b,img_y

def create_data_pairs(X,y,check):

  img_a = []
  img_b = []  
  img_y = []
  
  # our objective is to make half of the dataset with the same classes and half with different classes.

  # same class pairs for both image and audio
  img_a,img_b,img_y = create_same_class_pairs(X,y,check,img_a,img_b,img_y,same_class_list,4000)

  # different classes for both image and audio
  img_a,img_b,img_y = create_diff_class_pairs(X,y,img_a,img_b,img_y,2000)

  #first same class pairs for image but different class pairs for audio 
  #then different class pairs for image but same class pairs for audio

  if check==0 : #this block executed for image (check = 0)

    img_a,img_b,img_y = create_same_class_pairs(X,y,check,img_a,img_b,img_y,same_class_list1,1000) 
    img_a,img_b,img_y = create_diff_class_pairs(X,y,img_a,img_b,img_y,1000)

  else : #this block executed for audio (check = 1)

    img_a,img_b,img_y = create_diff_class_pairs(X,y,img_a,img_b,img_y,1000)
    img_a,img_b,img_y = create_same_class_pairs(X,y,check,img_a,img_b,img_y,same_class_list1,1000)

  
  return img_a,img_b,img_y

# create the final data to be fed into the siamese network for training

def multimodal_data_generate(X_img , y_img , X_aud , y_aud):

  img_a , img_b , img_y = create_data_pairs(X_img,y_img,0)
  aud_a , aud_b , aud_y = create_data_pairs(X_aud,y_aud,1)

  labels = []

  for i in range(len(img_a)):
    if(img_y[i]==-1 or aud_y[i] == -1):     # different class pair for both audio and image
      labels.append(0)
    elif(img_y[i]==aud_y[i]):     # same class pair for both audio and image
      labels.append(1)
    else:                     #same class pair for one and different class pair for another
      labels.append(0)
  
  return img_a,aud_a,img_b,aud_b,labels