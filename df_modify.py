import pandas as pd
import pickle

def pickle_dump(obj,path):
    with open(path,mode='wb') as f:
        pickle.dump(obj,f)

if __name__=='__main__':
    train_df=pd.read_csv('./input/train.csv')
    train_df=train_df[~train_df['EncodedPixels'].isnull()]
    train_df['Image']=train_df['Image_Label'].map(lambda x:x.split('_')[0])
    train_df['Class']=train_df['Image_Label'].map(lambda x:x.split('_')[1])
    classes=train_df['Class'].unique()
    train_df=train_df.groupby('Image')['Class'].agg(set).reset_index()
    for class_name in classes:
        train_df[class_name]=train_df['Class'].map(lambda x:1 if class_name in x else 0)

    img_2_ohe_vector={img:vec for img,vec in zip(train_df['Image'],train_df.iloc[:,2:].values)}

    train_df.to_csv('./input/train_label.csv')
    pickle_dump(img_2_ohe_vector,'./input/vector')
