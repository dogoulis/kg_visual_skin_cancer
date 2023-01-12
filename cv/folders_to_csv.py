import os
import pandas as pd
from imutils import paths
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class CreateFile():
    
    def __init__(self, folders, split_list, path_to_save):
        
        self.folders = folders # Dictionary with folder_path:folder_label
        self.path_to_save = path_to_save # list with the full-paths to save the csv
        self.split_list = split_list # list with the sizes to split the dataframes. 0 -> test_size, 1 -> train_size


    def images_to_csv(self, flag=False):
        
        if flag:
            pass

        else:
            data_frames = [] # list that consists of all the dataframes

            # first turn images to pd.Dataframe:
            for folder_path, folder_label in self.folders.items():
                df = self._images_to_df(path=folder_path, label_number=folder_label)
                data_frames.append(df)

            final_df = self._concat_df(data_frames) # then concatenate the dataframes in to a single one

            train_csv, val_csv, test_csv = self._train_test_split(final_df, \
                        test_size=self.split_list[0], train_size=self.split_list[1]) # then split in to training validation and test sets:

            # check if path exists:
            if not os.path.exists(self.path_to_save):
                os.makedirs(self.path_to_save)

            train_csv.to_csv(os.path.join(self.path_to_save, 'train.csv'))
            val_csv.to_csv(os.path.join(self.path_to_save, 'val.csv'))
            test_csv.to_csv(os.path.join(self.path_to_save, 'test.csv')) # then save the csv    

    def _images_to_df(self, path, label_number):

        image_paths = list(paths.list_images(path))  # getting all the image paths
        df = pd.DataFrame() 
        for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc='Transforming folder to Dataframe.'):
            # 0 for real, 1 for fake
            label = label_number
            df.loc[i, 'image_path'] = image_path
            df.loc[i, 'label'] = label
        return df

    def _concat_df(self, dfs):
        return pd.concat(dfs, axis=0)

    def _train_test_split(self, df, test_size, train_size):
        train_df, test_df = train_test_split(df, test_size=test_size)
        train_df, val_df = train_test_split(train_df, train_size=train_size)
        return train_df, val_df, test_df
