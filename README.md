# MRI-Ischemic-Lesion-Seg-UNet

##  CZ
* Tento repozitář byl vytvořen pro uložení zdrojových kódů vytvořených v programovacím jazyce Python s využitím 
  knihovny PyTorch za účelem segmentace ischemických ložisek v T1 vážených MRI datech s využitím konvolučních 
  neuronových sítí U-Net a také pro uložení hotových modelů tak, aby byly dostupné pro vyzkoušení detekce na novém
  MRI scanu.
  
* Pro tento projekt byla využita data z datasetu ATLAS R2.0 [1].

* Tento projekt byl vytvořen jako semestrální práce na Fakultě elektrotechniky a komunikačních technologií na Vysokém
  učení technickém v Brně. 

* Text této práce je také obsahem repozitáře - Ondrej_Nantl_SP.pdf - pouze česky.

* Pro spuštění kódu z tohoto repozitáře je nutné mít nainstalované knihovny PyTorch, nibabel, numpy, matplotlib, tqdm.

* Pro použití modelů (soubory s příponou .pb) je nutné je načíst pomocí torch.load.

* Kód využitý pro trénink a testování modelů je převážně obsažen v Jupyter Noteboocích ATLAStrain3D a ATLAStrain2D.

* Architektury navržených modelů jsou patrné z jejich .py skriptů - unet, unet3d, resunet a res3dunet.

## EN
* This repository was created for storing code created in Python using PyTorch used for segmentation of ischemic 
  lesions in T1W MRI data using CNN U-Net and for storing final models used for meta-analysis on new MRI scan.
  
* This project was created using data from ATLAS R2.0 dataset [1].

* This project was created as a bachelor thesis at the Faculty of Electrical Engineering and Communication at Brno
  University of Technology.

* Text of the thesis is also a part of this repository - Ondrej_Nantl_SP.pdf - only in Czech.

* For running code from this repository you need to install PyTorch, nibabel, numpy, matplotlib, tqdm.

* For usage the models (files with .pb suffix) shall be loaded using torch.load.

* Code used for training and testing of the models is mainly contained in Jupyter Notebooks ATLAStrain3D and ATLAStrain2D.

* Architectures of developed models are apparent from their .py skripts - unet, unet3d, resunet a res3dunet.
--------
[1] LIEW, Sook-Lei, Bethany P. LO, Miranda R. DONNELLY, et al. A large, curated, open-source stroke neuroimaging dataset to improve lesion segmentation algorithms. Scientific Data. 2022, 9(1). ISSN 2052-4463. Available at: https://doi.org/10.1038/s41597-022-01401-7
