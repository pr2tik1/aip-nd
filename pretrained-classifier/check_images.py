import os
import argparse
from classifier import classifier 
from time import time, sleep

from calculates_results_stats import calculates_results_stats
from print_results import print_results

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str, default= 'pet_images/', help='path to open images')
    parser.add_argument('--arch', default= 'vgg', help = 'select the architecture from vgg, alexnet, resnet')
    parser.add_argument('--dogfile', default= 'dognames.txt', help = 'selectt the text file')
    return parser.parse_args()

def get_pet_labels(image_dir):
    in_files = os.listdir(image_dir) 
    results_dic = dict()
    
    for idx in range(0, len(in_files),1):
        if in_files[idx][0] != ".":
            pet_label = ""
            file_names =  os.path.splitext(in_files[idx])
            #print(file_names)
            file_name = file_names[0].split("_")
            for word in file_name:
                if word.isalpha():
                    pet_label += word + " "
            
            pet_label = pet_label.strip().lower()
            #print( "\nFilename=", file_name, " Label=", pet_label)
            
            if in_files[idx] not in results_dic:
                results_dic[in_files[idx]] = [pet_label]
            else:
                print("** Warning: Duplicate files exist in directory:",in_files[idx], 
                        "with value: ", results_dic[in_files[idx]])

    return results_dic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

def classify_images(images_dir, results_dic, model):    
    for key in results_dic:
        train_label = classifier(images_dir + key, model)
        train_label = train_label.lower().strip()
        labels = results_dic[key][0]
        if labels in train_label:
            results_dic[key].extend((train_label, 1))
        else:
            results_dic[key].extend((train_label, 0))
     

def adjust_results4_isadog(results_dic, dogfile):
    dognames_dic = dict()

    with open(dogfile, "r") as infile:
        line = infile.readline()

        while line != "":
            line = line.rstrip()
            if line not in dognames_dic:
                dognames_dic[line]=[1]
                line = infile.readline()
            
    for key in results_dic:
        if results_dic[key][0] in dognames_dic:
            if results_dic[key][1] in dognames_dic:
                results_dic[key].extend((1,1))
            else:
                results_dic[key].extend((1,0))
        else:
            if results_dic[key][0] in dognames_dic:
                results_dic[key].extend((0,1))
            else:
                results_dic[key].extend((0,0))


def main():
    start_time = time()
    in_arg = get_input_args()
    results = get_pet_labels(in_arg.dir)
    
    classify_images(in_arg.dir , results, in_arg.arch)
    adjust_results4_isadog(results, in_arg.dogfile)

    results_stats = calculates_results_stats(results)
    print_results(results, results_stats, in_arg.arch, True, True)

    end_time = time()
    tot_time = end_time - start_time 
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    

if __name__ == "__main__":
    main()
