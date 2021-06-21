


    
def recall(y_true, y_pred):
    y_pred = K.sigmoid(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall



def precision(y_true, y_pred):
    y_pred = K.sigmoid(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec )/(prec +rec +K.epsilon()))

    
def sig_c_loss(y_true, y_pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels = y_true, logits = y_pred)

  

def sigmoid(x):
    return 1. / (1.  + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def normalize(image):
    return image / 255.


def parse_annotation_xml(ann_dir, img_dir, labels=[]):
    # This parser is utilized on VOC dataset
    all_imgs = []
    seen_labels = {}

    ann_files = os.listdir(ann_dir)
    for ann in tqdm(sorted(ann_files)):
        img = {'object': []}

        tree = et.parse(os.path.join(ann_dir, ann))

        duplicate = True
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = os.path.join(img_dir, elem.text)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag:#or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                     if 'name' in attr.tag: 
                        if attr.text == 'person':
                            duplicate = False
                            
                label_count = 1
                if duplicate is True:
                    label_count = 1
                    
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        
                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += label_count
                        else:
                            seen_labels[obj['name']] = label_count
                            
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]       

        if len(img['object']) > 0:
            all_imgs += [img] * label_count

        #if duplicate is True:
        #     all_imgs += [img] 
    return all_imgs, seen_labels

 
class BatchGenerator(Sequence):
    def __init__(self, images,config, shuffle=True,jitter=True,norm=None,callback=None):

       
        
        self._config = config

        self._shuffle = shuffle
        self._jitter = jitter
        self._norm = norm
        self._callback = callback

        if jitter:
            self._images = []
            for img in images:
                dup = True
                for obj in img['object']: 
                    if(obj['name'] == 'person'):
                        dup = False
                
                labelCount = 1
                if dup is True:
                    labelCount = 4
                
            
                self._images += [img]*labelCount
              
                
        else:
            self._images = images
       
    
        
        
            
    
       
        
        self.labels = self._config['LABELS']
        
        # augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.75, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self._aug_pipe = iaa.Sequential(
            [
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5), # add gaussian noise
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.1),  # vertically flip 20% of all images
                sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, per axis
                    translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},  # translate by -20 to +20 percent
                    rotate=(-5, 5),  # rotate by -45 to +45 degrees
                    shear=(-5, 5),  # shear by -16 to +16 degrees
                    # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 3),
                           [
                               # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                               iaa.OneOf([
                                   iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5), # add gaussian noise
                                   #iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means (kernel sizes between 2 and 7)
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians (kernel sizes between 2 and 7)
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                               # search either for all edges or for directed edges
                               # sometimes(iaa.OneOf([
                               #    iaa.EdgeDetect(alpha=(0, 0.7)),
                               #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                               # ])),
                               
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               iaa.Invert(0.05, per_channel=True), # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),  # change brightness of images
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

        if shuffle:
            np.random.shuffle(self._images)

    def __len__(self):
        return int(np.ceil(float(len(self._images)) / self._config['BATCH_SIZE']))

    def num_classes(self):
        return len(self._config['LABELS'])
    
    def has_label(self, label):
        return True
    
    def size(self):
        return len(self._images)

    def load_annotation(self, i):
        annots = []

        for obj in self._images[i]['object']:
            annot = [self._config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0:
            annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        if self._config['IMAGE_C'] == 1:
            image = cv2.imread(self._images[i]['filename'], cv2.IMREAD_GRAYSCALE)
            image = image[..., np.newaxis]
        elif self._config['IMAGE_C'] == 3:
            image = cv2.imread(self._images[i]['filename'])
        else:
            raise ValueError("Invalid number of image channels.")
        return image

    def __getitem__(self, idx):
        return self.getBatchNum(idx)
    
    def getBatchNum(self, idx):
        l_bound = idx * self._config['BATCH_SIZE']
        r_bound = (idx + 1) * self._config['BATCH_SIZE']

        if r_bound > len(self._images):
            r_bound = len(self._images)
            l_bound = r_bound - self._config['BATCH_SIZE']

        instance_count = 0
        if self._config['IMAGE_C'] == 3:
            x_batch = np.zeros((r_bound - l_bound,3, self._config['IMAGE_H'], self._config['IMAGE_W']))  # input images
        else:
            x_batch = np.zeros((r_bound - l_bound, 1, self._config['IMAGE_H'], self._config['IMAGE_W']))

        y_batch = np.zeros((r_bound - l_bound, len(self._config['LABELS'])))  # desired network output

        for train_instance in self._images[l_bound:r_bound]:
            # augment input image and fi x object's position and size
            img, all_objs = self.aug_image(train_instance, jitter=self._jitter)
            
            for obj in all_objs:
                if obj['name'] in self._config['LABELS']:
                    obj_indx = self._config['LABELS'].index(obj['name'])
                    y_batch[instance_count, obj_indx] = 1.0
                    
                    
            if self._norm is not None:
                img = self._norm(img)
            

            img = np.moveaxis(img, -1, 0)
            x_batch[instance_count] = img
            # increase instance counter in current batch
            instance_count += 1
        
        return x_batch, y_batch

    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self._images)

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        if self._config['IMAGE_C'] == 1:
            image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        elif self._config['IMAGE_C'] == 3:
            image = cv2.imread(image_name)
        else:
            raise ValueError("Invalid number of image channels.")

        if image is None:
            print('Cannot find ', image_name)
        if self._callback is not None:
            image, train_instance = self._callback(image, train_instance)

        h = image.shape[0]
        w = image.shape[1]
        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            image = self._aug_pipe(image=image)
           

        # resize the image to standard size
        image = cv2.resize(image, (self._config['IMAGE_W'], self._config['IMAGE_H']))
        
        if self._config['IMAGE_C'] == 1:
            image = image[..., np.newaxis]
        image = image[..., ::-1]  # make it RGB (it is important for normalization of some backends)

        return image, all_objs
