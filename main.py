from ML import *
from util import *
from sklearn.model_selection import train_test_split
from CNN import *
from sklearn.preprocessing import MinMaxScaler

# load all the images and the corresponding constants
# BRG - Grayscale - HSV
images, parameters, steady_state = load_images()
print('Images loaded...')

# split the data into a training set and a validation set
X_train, X_test, y_train, y_test = train_test_split(images, parameters, test_size=0.33, random_state=42)

scaler = MinMaxScaler()
scaler.fit(y_train)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)
y_test = [torch.tensor(p).float() for p in y_test]
y_train = [torch.tensor(p).float() for p in y_train]

train_data = list(zip(X_train, y_train))
valid_data = list(zip(X_test, y_test))


train(train_data=train_data, valid_data=valid_data, log_dir='\.', transform=None)
