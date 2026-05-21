Machine Learning — Convolutional Neural Network
Eksamensprojekt i Machine Learning med fokus på Convolutional Neural Networks (CNNs) til billedklassifikation.
Projektet sammenligner samme overordnede machine learning-workflow på to forskellige billeddatasæt:
Horses or Humans — binær klassifikation mellem hest og menneske
Flowers — multiklasse-klassifikation af fem blomstertyper
Formålet er at undersøge, hvordan en CNN-baseline kan bygges, evalueres og forbedres med hyperparameter-tuning, og hvordan modellens performance ændrer sig, når opgaven går fra en enkel binær klassifikation til en mere nuanceret multiklasse-opgave.
---
Projektinformation
Punkt	Information
Fagområde	Machine Learning
Emne	Convolutional Neural Networks
Uddannelsessted	Syddansk Erhvervsakademi
Hold	PBSWf26
Vejleder	Henrik Heinz Kühl
Studerende	Johan Hoppe, John Lorenzen, Casper Jensen
Afleveringsdato	22. maj 2026
---
Repository-struktur
```text
ML-Examen-2026-CNN/
├── models/
│   ├── Flowers.ipynb
│   └── Horses_or_Humans.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```
Notebooks ligger i `models/`, mens `README.md` og `requirements.txt` ligger i roden af projektet.
Når notebooks køres, kan der blive oprettet cache-filer og gemte modeller lokalt. Disse bruges til at undgå at træne modellerne forfra hver gang.
---
Datasæt
Horses or Humans
`Horses_or_Humans.ipynb` arbejder med Horses or Humans-datasættet fra TensorFlow Datasets.
Punkt	Værdi
Opgavetype	Binær billedklassifikation
Klasser	`Horse`, `Human`
Original billedstørrelse	300 × 300 × 3
Model-billedstørrelse	150 × 150 × 3
Train	719 billeder
Validation	154 billeder
Calibration	154 billeder
Test	256 billeder
Notebooken bruger et ekstra calibration-split til at undersøge threshold-valg for den tunede binære model.
Flowers
`Flowers.ipynb` arbejder med et Flowers-datasæt fra Kaggle via `kagglehub`.
Punkt	Værdi
Opgavetype	Multiklasse billedklassifikation
Klasser	`daisy`, `dandelion`, `roses`, `sunflowers`, `tulips`
Antal billeder	3.670
Model-billedstørrelse	180 × 180 × 3
Training split	80 %
Validation/test split	20 % delt i validation og test
Klassefordelingen vurderes i notebooken, og datasættet fremstår samlet set forholdsvis balanceret.
---
Metode
Begge notebooks følger samme overordnede kapitelstruktur:
Problemdefinition og datasæt-setup  
Datasæt indlæses, billedstørrelser defineres, seeds sættes, og TensorFlow pipelines oprettes.
Datasætanalyse og split-validering  
Klassenavne, billedfordelinger og splits undersøges for at sikre, at træning, validation og test bruges konsekvent.
Baseline CNN-model  
En simpel CNN bygges som referencepunkt. Modellen bruger convolutional layers, pooling, dense layers og passende output-lag til enten binær eller multiklasse-klassifikation.
Baseline-evaluering  
Modellen evalueres med test accuracy, test loss, confusion matrix samt precision, recall og F1-score.
Fine-tuning med Keras Tuner  
Hyperparametre optimeres i to trin:
en bred tuning-fase
en mere fokuseret refined tuning-fase omkring de bedste resultater
Konklusion  
Baseline og tunet model sammenlignes, og resultaterne fortolkes i forhold til datasættets sværhedsgrad.
---
Modeller og tuning
Projektet bruger to typer modeller:
Baseline CNN
Baseline-modellerne er simple CNN-arkitekturer, der fungerer som sammenligningsgrundlag. De trænes med callbacks som:
`EarlyStopping`
`ReduceLROnPlateau`
Formålet er at opnå en rimelig første model uden at optimere for mange hyperparametre manuelt.
Keras-tuned CNN
Den tunede model bygges med Keras Tuner. Search space inkluderer blandt andet:
data augmentation
antal convolutional blocks
antal filtre
kernel size
dropout
L2-regularisering
dense units
learning rate
Tuning-strategien er todelt, så projektet først finder et lovende område og derefter søger mere præcist omkring de bedste hyperparametre.
---
Resultater
Samlet modelperformance
Datasæt	Model	Test accuracy	Test loss
Horses or Humans	Baseline CNN	0.8516	1.6646
Horses or Humans	Keras-tuned CNN	0.8477*	0.5658
Flowers	Baseline CNN	0.6477	0.9460
Flowers	Keras-tuned CNN	0.8068	0.5766
* For Horses or Humans er den tunede accuracy angivet med validation-baseret threshold. Modellen opnåede samtidig en test AUC på 0.9839, men accuracy blev ikke tydeligt forbedret i forhold til baseline.
Fortolkning
På Flowers gav hyperparameter-tuning en tydelig forbedring: test accuracy steg fra 0.6477 til 0.8068, og test loss faldt fra 0.9460 til 0.5766.
På Horses or Humans klarede baseline-modellen sig allerede relativt godt med 0.8516 test accuracy. Den tunede model reducerede test loss og opnåede høj AUC, men forbedrede ikke den endelige accuracy væsentligt. Det viser, at tuning ikke altid giver en bedre samlet accuracy, især når baseline allerede finder et stærkt beslutningsmønster.
---
Installation
Projektet er lavet i Python og Jupyter Notebook.
1. Klon repository
```bash
git clone https://github.com/RazorSDU/ML-Examen-2026-CNN.git
cd ML-Examen-2026-CNN
```
2. Opret virtuelt miljø
Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```
3. Installer dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
Hvis `requirements.txt` giver problemer, kan pakkerne installeres direkte:
```bash
python -m pip install jupyter ipykernel numpy pandas matplotlib scikit-learn tensorflow tensorflow-datasets keras-tuner kagglehub
```
4. Start Jupyter
```bash
jupyter notebook
```
Åbn derefter notebooks i `models/`:
```text
models/Horses_or_Humans.ipynb
models/Flowers.ipynb
```
---
Kørselskrav
Projektet er testet i WSL/Ubuntu med GPU-accelereret TensorFlow.
Anbefalet setup:
Python 3.10+
NVIDIA GPU med CUDA-understøttelse
Minimum 16 GB RAM
Tilstrækkelig diskplads til datasæt, model-cache og tuner-resultater
Den samlede køretid kan ligge omkring 5–10 timer, afhængigt af GPU, batch size og om cache/tuning køres forfra.
> Bemærk: Notebooks indeholder enkelte miljøspecifikke celler til TensorFlow GPU-setup i WSL. Hvis projektet køres i Google Colab, Windows uden WSL eller et andet Linux-miljø, kan disse celler kræve tilpasning eller springes over.
---
Reproducerbarhed og cache
Begge notebooks bruger faste seeds for at gøre resultaterne mere reproducerbare:
```python
SEED = 42
```
Derudover gemmes modeller, træningshistorik og tuner-resultater lokalt i cache-mapper. Det gør det muligt at genåbne notebooks uden at skulle gentage hele trænings- og tuningprocessen.
Hvis modellerne skal trænes forfra, kan de relevante `FORCE_RETRAIN...` variabler ændres i notebooks.
---
Evaluering
Modellerne vurderes ikke kun på accuracy. Projektet bruger også:
confusion matrix
precision
recall
F1-score
test loss
AUC for den binære Horses or Humans-model
Det gør evalueringen mere robust, fordi accuracy alene kan skjule, om modellen klarer sig dårligt på enkelte klasser.
---
Konklusion
Projektet viser, hvordan CNNs kan anvendes til billedklassifikation på både binære og multiklasse-problemer.
Den vigtigste forskel mellem de to datasæt er, at Flowers er en mere kompleks opgave med fem visuelt overlappende klasser, hvor tuning giver en markant forbedring. Horses or Humans er mere enkel som binær klassifikation, hvor baseline allerede klarer sig stærkt, og tuning derfor primært forbedrer loss og sandsynlighedsseparation frem for accuracy.
Samlet viser projektet, at et ensartet workflow med datasætanalyse, baseline, evaluering og systematisk tuning giver et godt grundlag for at forstå både modellens styrker og dens begrænsninger.
---
Teknologier
Python
Jupyter Notebook
TensorFlow / Keras
TensorFlow Datasets
Keras Tuner
KaggleHub
NumPy
Pandas
Matplotlib
scikit-learn
---
Mulige forbedringer
Fremtidige forbedringer kunne være:
transfer learning med f.eks. MobileNetV2, EfficientNet eller ResNet
mere systematisk data augmentation
class weighting ved mere ubalancerede datasæt
Grad-CAM eller anden forklarlig visualisering
eksport af modeller til inference
samlet script til træning uden for notebooks
bedre dokumentation af hardware, CUDA-version og træningstid pr. model
