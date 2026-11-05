const fs=require('fs');
const {Document,Packer,Paragraph,TextRun,ImageRun,Table,TableRow,TableCell,WidthType,AlignmentType,BorderStyle,ShadingType,Footer,PageNumber,SectionType,TabStopType}=require('docx');
const F='Times New Roman';
const run=(t,o={})=>new TextRun({text:t,font:F,size:o.size||20,bold:o.b,italics:o.i,smallCaps:o.sc});
const rich=(s,o={})=>s.split('*').map((x,i)=>run(x,{...o,i:i%2===1||o.i})).filter(r=>true);
const body=s=>new Paragraph({alignment:AlignmentType.JUSTIFIED,spacing:{after:80,line:240},indent:{firstLine:200},children:rich(s)});
const H=s=>new Paragraph({spacing:{before:140,after:60},keepNext:true,children:[run(s,{b:true})]});
const SH=s=>new Paragraph({spacing:{before:60,after:40},keepNext:true,children:[run(s,{i:true})]});
const eq=(t,n)=>new Paragraph({tabStops:[{type:TabStopType.CENTER,position:2000},{type:TabStopType.RIGHT,position:4004}],spacing:{after:80},children:[run('\t'+t,{i:true}),run('\t('+n+')')]});
const img=(f,w,h,cap)=>[new Paragraph({alignment:AlignmentType.CENTER,keepNext:true,spacing:{before:60,after:20},children:[new ImageRun({type:'png',data:fs.readFileSync(f),transformation:{width:w,height:h},altText:{title:cap,description:cap,name:cap}})]}),new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:100},children:[run(cap,{size:16})]})];
const bd={style:BorderStyle.SINGLE,size:4,color:'000000'}; const borders={top:bd,bottom:bd,left:bd,right:bd};
const table=(num,title,rows,widths)=>{
  const out=[new Paragraph({alignment:AlignmentType.CENTER,keepNext:true,spacing:{before:80,after:0},children:[run('TABLE '+num,{size:16,sc:true})]}),new Paragraph({alignment:AlignmentType.CENTER,keepNext:true,spacing:{after:40},children:[run(title,{size:16,sc:true})]})];
  const total=widths.reduce((a,b)=>a+b,0);
  out.push(new Table({width:{size:total,type:WidthType.DXA},columnWidths:widths,alignment:AlignmentType.CENTER,rows:rows.map((r,ri)=>new TableRow({cantSplit:true,children:r.map((c,ci)=>new TableCell({borders,width:{size:widths[ci],type:WidthType.DXA},margins:{top:30,bottom:30,left:50,right:50},shading:ri===0||r.length===1?{type:ShadingType.CLEAR,fill:'E7E6E6'}:undefined,columnSpan:r.length===1?widths.length:undefined,children:[new Paragraph({alignment:ci===0&&r.length>1?AlignmentType.LEFT:AlignmentType.CENTER,children:[run(c,{size:16,b:ri===0||r.length===1})]})]}))}))}));
  out.push(new Paragraph({spacing:{after:80},children:[]}));return out;};
const W5=[1404,650,650,650,650];
const title=[
 new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:120},children:[run('Deep Learning-Based Classification of Kidney Cyst, Stone and Tumor from CT Images: A Source-Aware Evaluation',{b:true,size:40})]}),
 new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:0},children:[run('Aaryaman Singh',{b:true})]}),
 new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:0},children:[run('Reg. No. 2430010383')]}),
 new Paragraph({alignment:AlignmentType.CENTER,children:[run('Department of Computer Science & Engineering (Data Science)')]}),
 new Paragraph({alignment:AlignmentType.CENTER,children:[run('Manipal University Jaipur, Jaipur, Rajasthan, India')]}),
 new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:160},children:[run('aaryamansingh366@gmail.com')]})];
const abs=`Automatic recognition of kidney cysts, stones and tumors in CT images can support radiologists, but reported accuracies are often inflated by correlated images appearing in both training and test data. This paper studies four-class kidney CT classification (cyst, normal, stone, tumor) on a public dataset of 2,935 images using a classical HOG + linear SVM baseline, a lightweight five-block convolutional neural network (CNN, 0.54 M parameters) and a ResNet-18 trained from scratch. We evaluate under two protocols: a block-wise split of consecutive slices and a stricter source-held-out split in which whole scan groups are excluded from training. Under the block split, all models perform very well (ResNet-18: 99.6% accuracy, 98.7% macro F1) and even the HOG baseline reaches 100%. Under the source-held-out split, performance drops sharply (ResNet-18: 69.6% accuracy, 60.4% macro F1; CNN: 57.1%; HOG + SVM: 19.6%). Our analysis shows that image dimensions are almost perfectly class-specific and that a pixel nearest-neighbour classifier scores 100% on the block split, indicating that high scores largely reflect scan-level shortcuts. We recommend patient- or scan-level splitting when benchmarking kidney CT classifiers.`;
const kw='Kidney CT, cyst, stone, tumor, deep learning, convolutional neural network, ResNet-18, data leakage, medical image classification';
const sec1=[
 body(`Kidney disease imposes a large clinical burden, and computed tomography (CT) is the main imaging modality for detecting renal cysts, stones and tumors. Interpreting these studies is time consuming and depends on the reader's experience, which motivates computer-aided diagnosis based on deep learning [1], [2]. Convolutional networks have achieved strong results across medical imaging tasks [3], and public kidney CT datasets now make four-class screening (cyst, normal, stone, tumor) possible [1].`),
 body(`A recurring weakness of such studies is evaluation. CT datasets typically contain many neighbouring slices from the same patient or scanner, so a random image-level split places near-duplicates in both training and test sets. Models can then exploit patient or acquisition-specific cues rather than disease features, a failure mode documented for chest radiographs [4], [5]. Reported accuracies near 100% should therefore be interpreted cautiously.`),
 body(`This work makes three contributions. (i) We benchmark a classical HOG + SVM baseline, a lightweight CNN and ResNet-18 on a four-class kidney CT dataset. (ii) We expose the dataset's source structure: image dimensions are almost class-specific, and trivial classifiers reach very high accuracy on a conventional split. (iii) We introduce a source-held-out evaluation, which excludes entire scan groups from training and gives a more conservative estimate of generalization.`),
 H('II. RELATED WORK'),
 body(`Deep learning surveys describe the rapid adoption of CNNs for classification, detection and segmentation in medical imaging [3]. Residual networks [6] and batch normalization [7] are now standard building blocks and are commonly used with transfer learning when labelled data are scarce. For kidney CT, Islam *et al.* [1] released a large dataset of cyst, normal, stone and tumor images and compared transfer-learning CNNs and vision transformers with explainability analysis, reporting very high accuracy on their test sets. Hand-crafted descriptors such as histograms of oriented gradients [8] with linear classifiers remain a useful low-cost baseline.`),
 body(`Concerns about generalization are well documented. Zech *et al.* [4] showed that a pneumonia classifier's performance varied markedly across hospitals, and DeGrave *et al.* [5] showed that COVID-19 radiograph classifiers can rely on shortcuts rather than pathology. These studies motivate the source-aware evaluation used here. To our knowledge, the effect of scan-source grouping on this kidney CT dataset has received limited attention, which is the gap this paper addresses.`),
 H('III. PROPOSED METHODOLOGY'),
 SH('A. System Architecture'),
 body(`Fig. 1 shows the pipeline. Each CT slice is resized, normalized and augmented before being passed to the classifier, which outputs one of four classes. Two data-splitting protocols (Section III-B) are applied identically to every model.`),
 ...img('fig_arch.png',262,188,'Fig. 1. Workflow and architecture of the lightweight CNN used in this study.'),
 SH('B. Dataset'),
 body(`The dataset contains 2,935 JPEG CT images in four class folders: Normal (1,425), Cyst (817), Tumor (512) and Stone (181) (Fig. 2). The classes are imbalanced, and Stone is the minority class. Original image sizes vary across 28 distinct dimensions (e.g., 512×512, 867×701, 1221×988), and images include axial and coronal views (Fig. 3).`),
 ...img('fig_dist.png',262,127,'Fig. 2. Class distribution of the dataset.'),
 ...img('fig_samples.png',262,141,'Fig. 3. Example CT slices per class (resized to 128×128).'),
 body(`*Source structure.* In our analysis, 27 of the 28 original image sizes occur in exactly one class. A classifier that predicts the majority class for each image size scores 69.2% on the block-split test set, and a pixel-space 1-nearest-neighbour classifier scores 100%. The dataset has no patient identifiers, so patient-level splitting is impossible; we instead use the two proxies below.`),
 body(`*Block split.* Within each class, images are ordered by file index, grouped into contiguous blocks of 10, and blocks are assigned to training, validation and test sets (about 70/15/15). This yields 2,044/440/451 images.`),
 body(`*Source-held-out split.* All images sharing a class and original image size are treated as one source group, and whole groups are assigned to a single subset. This yields 2,116/441/378 training/validation/test images (test: 92 Cyst, 195 Normal, 15 Stone, 76 Tumor). A random seed was chosen only so that every class appears in validation and test with adequate counts; no model performance was used for this choice.`),
 SH('C. Data Preprocessing'),
 body(`Images are converted to RGB, resized to 128×128 by bilinear interpolation, average-pooled to 64×64 for training efficiency, and standardized with the training-set channel mean and standard deviation. During training, each batch receives random horizontal flips, rotation (±10°), scaling (±10%), translation (±6%), and brightness/contrast jitter (±10%). No augmentation is applied at validation or test time.`),
 SH('D. Models'),
 body(`*HOG + SVM baseline.* Grayscale 128×128 images are described by HOG features (9 orientations, 16×16 cells, 2×2 blocks; 1,764 dimensions) [8], standardized, and classified with a class-balanced linear SVM whose regularization constant is selected on the validation set.`),
 body(`*Lightweight CNN.* The network (Fig. 1) has five blocks of 3×3 convolution, batch normalization and ReLU followed by 2×2 max pooling, with 32, 64, 128, 128 and 256 channels, then global average pooling, dropout (0.4) and a fully connected layer with softmax. It has 537,636 parameters.`),
 body(`*ResNet-18.* The standard ResNet-18 [6] with a four-way output layer (11,178,564 parameters) is trained from random initialization, because pretrained weights were not available in our environment.`),
 body(`Both networks minimize a class-weighted cross-entropy with label smoothing ε = 0.05:`),
 eq('L = −Σ w(c) · ỹ(c) · log p(c)','1'),
 body(`where p is the softmax output, ỹ the smoothed label and w(c) = [N/(4·n(c))]^½ the weight of class c, where n(c) is its number of training images and N the total. Evaluation metrics are`),
 eq('Acc = (TP+TN)/(TP+TN+FP+FN),  F1 = 2PR/(P+R)','2'),
 body(`with precision (P), recall (R) and F1 macro-averaged over the four classes; one-vs-rest macro AUC is also reported.`),
 H('IV. EXPERIMENTAL SETUP'),
 body(`All experiments ran on a single CPU core (3 GB RAM, no GPU) using Python 3, PyTorch [9], scikit-learn and scikit-image. Networks were trained with AdamW [10] (weight decay 10⁻⁴; peak learning rate 2×10⁻³ for the CNN and 10⁻³ for ResNet-18), a one-cycle schedule [11] and batch size 32. The CNN was trained for 10 epochs and ResNet-18 for 6 epochs; the checkpoint with the highest validation macro F1 was evaluated once on the test set. Each configuration was run once with a fixed seed, so no confidence intervals are available. Because of the compute budget, input resolution was limited to 64×64 pixels and training times were short (about 200–280 s per model).`),
 H('V. RESULTS AND DISCUSSION'),
 SH('A. Block Split'),
 body(`Table I reports test results. All three models score very high; HOG + SVM classifies all 451 test images correctly and ResNet-18 reaches 99.6% accuracy (2 Cyst images misclassified as Stone). The CNN reaches 96.9%; its errors are 11 Cyst images predicted as Stone and 3 as Tumor.`),
 ...table('I','Test results, block split (n = 451)',[['Model','Acc.','Prec.','Rec.','F1'],['HOG + SVM','100.0','100.0','100.0','100.0'],['Light CNN','96.9','90.5','97.3','92.9'],['ResNet-18','99.6','97.8','99.6','98.7']],W5),
 SH('B. Source-Held-Out Split'),
 body(`Table II shows that performance drops substantially when whole source groups are held out. The HOG baseline falls to 19.6% accuracy, below the 51.6% obtained by always predicting Normal, showing that its block-split result relied on source-specific appearance. ResNet-18 is the best model (69.6% accuracy, 60.4% macro F1, AUC 0.763), followed by the CNN (57.1%, 41.1%, AUC 0.695).`),
 ...table('II','Test results, source-held-out split (n = 378)',[['Model','Acc.','Prec.','Rec.','F1'],['HOG + SVM','19.6','24.4','22.6','18.0'],['Light CNN','57.1','37.3','54.7','41.1'],['ResNet-18','69.6','68.9','71.3','60.4']],W5),
 body(`Per-class recall (Table III) shows that Tumor is the hardest class: the CNN recovers none of the 76 tumor images, and ResNet-18 recovers 32.9%, mostly confusing them with Normal (Fig. 4). Stone recall is high for ResNet-18 (all 15 images) but the Stone test set is very small, so this estimate is unreliable.`),
 ...table('III','Per-class recall (%), source-held-out split',[['Model','Cyst','Normal','Stone','Tumor'],['HOG + SVM','38.0','16.4','33.3','2.6'],['Light CNN','100.0','59.0','60.0','0.0'],['ResNet-18','71.7','80.5','100.0','32.9']],W5),
 ...img('fig_cm.png',262,143,'Fig. 4. ResNet-18 confusion matrices (counts) on the two test sets.'),
 ...table('IV','Model size, AUC and CPU inference time',[['Model','Params','AUC (block)','AUC (held-out)','ms/img'],['Light CNN','0.54 M','0.999','0.695','2.7'],['ResNet-18','11.2 M','1.000','0.763','4.2']],[1004,700,800,900,600]),
 SH('C. Discussion'),
 body(`Training accuracy approaches 100% for both networks while held-out validation accuracy stays at roughly 45–73% after the first few epochs (Fig. 5), a large generalization gap and a sign of overfitting to source-specific cues. The sharp difference between Tables I and II, together with the near-perfect size-based and pixel-based shortcuts, suggests that block-split scores mainly measure how well a model recognizes scans seen during training. The source-held-out figures are a more conservative, though imperfect, estimate, since holding out an image-size group is only a proxy for holding out patients.`),
 ...img('fig_curves.png',262,142,'Fig. 5. Training and held-out validation accuracy (source-held-out split).'),
 body(`*Comparison with prior work.* Islam *et al.* [1] report accuracies above 99% with transfer-learning models on this type of data. Our block-split results (96.9–100%) are consistent with this range, but we caution that such numbers may not transfer to unseen scanners or patients. We did not reproduce their pipeline, so direct comparison is limited.`),
 body(`*Limitations.* (i) Results come from single runs without confidence intervals. (ii) Resolution is low (64×64) and training is short. (iii) ResNet-18 is trained from scratch rather than fine-tuned from pretrained weights. (iv) The Stone test set has only 15 images in the held-out split. (v) The held-out split is a proxy for patient-level separation because patient identifiers are unavailable. (vi) Metrics were computed only on 2D images, not clinically validated.`),
 H('VI. CONCLUSION AND FUTURE WORK'),
 body(`We compared a HOG + SVM baseline, a lightweight CNN and ResNet-18 for four-class kidney CT classification under two evaluation protocols. Under a conventional block split every model scores very high, but under a source-held-out split accuracy falls to 19.6–69.6%, with ResNet-18 the strongest and tumor recognition the weakest. The dataset's strong class-to-source correlation means conventional splits can overstate real performance, and we recommend patient- or scan-level partitioning for future benchmarks.`),
 body(`Future work includes fine-tuning pretrained networks at higher resolution, obtaining patient identifiers for true patient-level splits, repeating experiments over multiple seeds, applying CT windowing and explainability methods such as Grad-CAM, and validating on external datasets.`),
 H('ACKNOWLEDGMENT'),
 body(`The author thanks Manipal University Jaipur for academic support, and the creators of the public kidney CT dataset used in this study.`),
 H('REFERENCES')];
const refs=[
 'M. N. Islam, M. Hasan, M. K. Hossain, M. G. R. Alam, M. Z. Uddin, and A. Soylu, “Vision transformer and explainable transfer learning models for auto detection of kidney cyst, stone and tumor from CT-radiography,” Sci. Rep., vol. 12, Art. no. 11440, 2022.',
 'A. Esteva et al., “Deep learning-enabled medical computer vision,” npj Digit. Med., vol. 4, Art. no. 5, 2021.',
 'G. Litjens et al., “A survey on deep learning in medical image analysis,” Med. Image Anal., vol. 42, pp. 60–88, 2017.',
 'J. R. Zech et al., “Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study,” PLoS Med., vol. 15, no. 11, e1002683, 2018.',
 'A. J. DeGrave, J. D. Janizek, and S.-I. Lee, “AI for radiographic COVID-19 detection selects shortcuts over signal,” Nat. Mach. Intell., vol. 3, pp. 610–619, 2021.',
 'K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778.',
 'S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in Proc. Int. Conf. Mach. Learn. (ICML), 2015, pp. 448–456.',
 'N. Dalal and B. Triggs, “Histograms of oriented gradients for human detection,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2005, pp. 886–893.',
 'A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2019, pp. 8024–8035.',
 'I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019.',
 'L. N. Smith and N. Topin, “Super-convergence: Very fast training of neural networks using a large learning rate,” arXiv:1708.07120, 2017.'];
const refP=refs.map((r,i)=>new Paragraph({alignment:AlignmentType.JUSTIFIED,spacing:{after:50,line:240},indent:{left:300,hanging:300},children:[run('['+(i+1)+']\t'+r,{size:18})],tabStops:[{type:TabStopType.LEFT,position:300}]}));
const page={size:{width:12240,height:15840},margin:{top:936,right:936,bottom:936,left:936,header:720,footer:720}};
const footer=new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,children:[new TextRun({children:[PageNumber.CURRENT],font:F,size:16})]})]});
const absP=new Paragraph({alignment:AlignmentType.JUSTIFIED,spacing:{after:80,line:240},children:[run('Abstract—',{b:true,i:true}),run(abs,{b:true})]});
const kwP=new Paragraph({alignment:AlignmentType.JUSTIFIED,spacing:{after:120,line:240},children:[run('Keywords—',{b:true,i:true}),run(kw,{i:true})]});
const doc=new Document({creator:'Aaryaman Singh',title:'Deep Learning-Based Classification of Kidney Cyst, Stone and Tumor from CT Images',styles:{default:{document:{run:{font:F,size:20}}}},sections:[
 {properties:{page},footers:{default:footer},children:title},
 {properties:{type:SectionType.CONTINUOUS,page,column:{count:2,space:360}},footers:{default:footer},children:[absP,kwP,H('I. INTRODUCTION'),...sec1,...refP]}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync('Kidney_CT_IEEE_Research_Paper.docx',b);console.log('ok')});
