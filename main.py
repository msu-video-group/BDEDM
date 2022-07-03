import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Detects bit-depth enhancement')
parser.add_argument('--dataset', type=Path, 
                    help='Folders containing videos to predict true bit-depth')
parser.add_argument('--action', type='str', default='test',
                    help='Select train or test')
parser.add_argument('--usecase', type=str, default='compressed',
                    help='compressed or uncompressed (compressed)')                    
parser.add_argument('--bit_depth', type=int, default=10, 
                    help='Videos bit-depth 8 or 10 (10)')
parser.add_argument('--preset', type=str, default='fast', 
                    help='Model speed preset fast or precise (fast)')
parser.add_argument('--modelSVM', type=Path, 
                    help='SVM model path')
parser.add_argument('--modelGBM', type=Path, 
                    help='GBM model path')
parser.add_argument('--modelRandomForest', type=Path, 
                    help='RandomForest model path')
parser.add_argument('--gt', type=str, default='GT', 
                    help='GT folder name (GT)')
parser.add_argument('--W', type=int, default=1920, 
                    help='Video width (1920)')
parser.add_argument('--H', type=int, default=1080, 
                    help='Video height (1080)')
parser.add_argument('--multiframe', type=bool, default=False, 
                    help='For uncompressed video use more than one frame (False)')
                    
args = parser.parse_args()

if args.action == 'train':
    if args.action == 'uncompressed':
        X, y = calcUncompr(args.dataset, GTFolder=args.gt, targetBitDepth=args.bit_depth, \
                            multiframe=args.multiframe, videoMetaInfo={'W' : arg.W, 'H' : args.H})
                            
        modelSVM = make_pipeline(StandardScaler(), SVC(probability=True, class_weight='balanced'))
        modelSVM.fit(X, y)
        modelGBM = make_pipeline(StandardScaler(), lgb.LGBMClassifier(class_weight='balanced'))
        modelGBM.fit(X, y)
        pickle.dump(modelSVM, open(args.modelSVM, 'wb'))
        pickle.dump(modelGBM, open(args.modelGBM, 'wb'))
    
    elif args.action == 'compressed':
        X, y = calcCompr(args.dataset, modelSVMpath=args.modelSVM, modelGBMpath=args.modelGBM, \
                         preset=args.preset, GTFolder=args.gt, targetBitDepth=args.bit_depth)
        modelRF = RandomForestClassifier(class_weight='balanced')
        modelRF.fit(X, y)
        pickle.dump(modelRF, open(args.modelRandomForest, 'wb'))

    else:
        print('Err: Unknown usecase')
elif args.action == 'test':
    if args.action == 'uncompressed':
        X, y = calcUncompr(args.dataset, GTFolder=args.gt, targetBitDepth=args.bit_depth, \
                            multiframe=args.multiframe, videoMetaInfo={'W' : arg.W, 'H' : args.H})
        modelSVM = pickle.load(open(args.modelSVM, 'rb'))
        modelGBM = pickle.load(open(args.modelGBM, 'rb'))

        y_proba = (modelSVM.predict_proba(X)[:,0] + modelGBM.predict_proba(X)[:,0]) / 2
        y_pred = np.rint(y_proba)  
        print(y_pred)
        
    elif args.action == 'compressed':
        X, y = calcCompr(args.dataset, modelSVMpath=args.modelSVM, modelGBMpath=args.modelGBM, \
                         preset=args.preset, GTFolder=args.gt, targetBitDepth=args.bit_depth)
        modelRF = pickle.load(open(args.modelRandomForest, 'rb'))
        y_pred = modelRF.predict(X)
        print(y_pred)
    else:
        print('Err: Unknown usecase')
else:
    print('Err: Unknown action')