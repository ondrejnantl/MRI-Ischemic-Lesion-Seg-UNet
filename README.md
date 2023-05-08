# MRI-Ischemic-Lesion-Seg-UNet

##  CZ
* Tento repozitář byl vytvořen pro uložení zdrojových kódů vytvořených v programovacím jazyce Python s využitím 
  knihovny PyTorch za účelem segmentace ischemických ložisek v T1 vážených MRI datech s využitím konvolučních 
  neuronových sítí U-Net a také pro uložení hotových modelů tak, aby byly dostupné pro vyzkoušení detekce na novém
  MRI scanu.
  
* Pro tento projekt byla využita data z datasetu ATLAS R2.0 [1].

* Tento projekt byl vytvořen jako diplomová práce na Fakultě elektrotechniky a komunikačních technologií na Vysokém
  učení technickém v Brně. 

* Text této práce je také obsahem repozitáře - Ondrej_Nantl_DP.pdf - pouze česky.

* Navržené natrénované modely je možné stáhnout zde: https://drive.google.com/file/d/1_2RTypdlTrKZINnVOGDk60jkpyqMSOLH/view?usp=sharing

* Pro spuštění kódů z tohoto repozitáře je nutné mít nainstalované knihovny podle requirements.txt.

* Pro použití vlastních navržených modelů (soubory s příponou .pb) je určen skript inference volatelný přes terminál. 
  Bližší informace k volání jsou dostupné při zavolání python inference.py -h v terminálu.

* Modely navržené s využitím nnUNet lze použít po extrahování standardním postupem pro inferenci uvedenou v dokumentaci
  nnUNet (více zde: https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1#run-inference).

* Kód využitý pro trénink a testování modelů je převážně obsažen ve skriptech trainMSD a trainResMSD.

* Architektury navržených modelů jsou patrné z jejich .py skriptů - unet3d a res3dunet.

* Další skripty obsahují kód pro optimalizaci hyperparametrů - optimalization and optimalizationRes.

* Výpočet výkonnostních metrik provádí skript evalMetrics.

* Dále repozitář obsahuje skripty pro vykreslení grafů a ukázek segmentací - compareParameters, createBoxplots,
  plotAllNets, plotAndSaveMetrics, scatterPlots.

* Repozitář také obsahuje skripty s podpůrnými kódy - BIDS2MSD, convertPreds, cropAndResampData, loaders, 
  loss_fcns, transforms.

## EN
* This repository was created for storing code created in Python using PyTorch used for segmentation of ischemic 
  lesions in T1W MRI data using CNN U-Net and for storing final models used for meta-analysis on new MRI scan.
  
* This project was created using data from ATLAS R2.0 dataset [1].

* This project was created as a diploma thesis at the Faculty of Electrical Engineering and Communication at Brno
  University of Technology.

* Text of the thesis is also a part of this repository - Ondrej_Nantl_DP.pdf - only in Czech.

* Created trained models can be downloaded from: https://drive.google.com/file/d/1_2RTypdlTrKZINnVOGDk60jkpyqMSOLH/view?usp=sharing

* For running all code from this repository you need to install libraries listed in requirements.txt.

* For usage the own created models (files with .pb suffix) there is a script inference callable from terminal. 
  For more information run python inference.py -h in terminal.

* Models created using nnUNet can be used after extraction from the .zip archive using standard inference procedure 
  described in nnUNet documentation (more over here: https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1#run-inference).

* Code used for training and testing of the models is mainly contained in scripts trainMSD a trainResMSD.

* Architectures of developed models are apparent from their .py skripts - unet3d and res3dunet.

* Other scripts contains code for optimalization of hyperparameters - optimalization and optimalizationRes.

* The evaluation is performed using script evalMetrics.

* Further more the repository contains scripts for plotting performance plots a examples of 
  segmentation - compareParameters, createBoxplots, plotAllNets, plotAndSaveMetrics, scatterPlots.

* There are also scripts with supplementary code - BIDS2MSD, convertPreds, cropAndResampData, loaders, 
  loss_fcns, transforms. 
--------
[1] LIEW, Sook-Lei, Bethany P. LO, Miranda R. DONNELLY, et al. A large, curated, open-source stroke neuroimaging dataset to improve lesion segmentation algorithms. Scientific Data. 2022, 9(1). ISSN 2052-4463. Available at: https://doi.org/10.1038/s41597-022-01401-7
