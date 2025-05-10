# 🎮 Aprenentatge per Reforç - CliffWalking

Aquest repositori conté implementacions de diferents algorismes d'aprenentatge per reforç per resoldre l'entorn CliffWalking de Gymnasium. Els algorismes implementats són:

- **🧮 Iteració de Valor (Value Iteration)**: Algorisme basat en programació dinàmica que troba la política òptima de manera iterativa.
- **📊 Estimació Directa (Direct Estimation)**: Algorisme basat en el model que aprèn les probabilitats de transició i les recompenses directament de l'experiència.
- **🔍 Q-Learning**: Algorisme sense model (model-free) que aprèn els valors Q per a cada parell estat-acció mitjançant actualitzacions incrementals.
- **🚀 REINFORCE**: Algorisme de gradient de política que optimitza directament la política sense aprendre explícitament els valors Q.


## 📥 Instal·lació

### Requisits

Per executar aquest codi necessitaràs Python 3.7 o superior i les següents llibreries:

```bash
pip install gymnasium numpy matplotlib pandas tqdm seaborn
```

### Verificació d'instal·lació

Pots verificar que l'entorn CliffWalking està correctament instal·lat executant:

```python
import gymnasium as gym
env = gym.make('CliffWalking-v0', is_slippery=True)
print(f"Nombre d'estats: {env.observation_space.n}")
print(f"Nombre d'accions: {env.action_space.n}")
```

## 📂 Estructura del Repositori

```
.
├── value_iteration.py      # Implementació d'Iteració de Valor
├── direct_estimate.py      # Implementació d'Estimació Directa
├── q_learning.py           # Implementació de Q-Learning
├── reinforce.py            # Implementació de REINFORCE
└── README.md               # Aquest fitxer
```

## 🚀 Instruccions d'Execució i Parametrització

### 🧮 Iteració de Valor (`value_iteration.py`)

<div style="background-color: #f0f9eb; padding: 10px; border-left: 5px solid #52c41a; margin-bottom: 15px;">
Aquest algorisme calcula iterativament la funció de valor i deriva una política òptima.
</div>

**Execució:**
```bash
python value_iteration.py
```

**Mode d'ús:**
1. Escull una opció:
   - `1`: Executar experiments complets (amb diferents combinacions de paràmetres)
   - `2`: Visualitzar resultats existents (si ja has executat experiments prèviament)
   - `3`: Executar un únic experiment amb gamma=0.99

**Paràmetres configurables:**
- `gamma`: Factor de descompte (valor entre 0 i 1). Valors més propers a 1 donen més importància a les recompenses futures.
- `theta`: Llindar de convergència per aturar l'algorisme.
- `custom_reward`: Funció personalitzada per modificar les recompenses de l'entorn.

### 📊 Estimació Directa (`direct_estimate.py`)

<div style="background-color: #f6ffed; padding: 10px; border-left: 5px solid #73d13d; margin-bottom: 15px;">
Aquest algorisme aprèn un model de transició i recompensa a partir de l'experiència, i després utilitza aquest model per calcular la política òptima.
</div>

**Execució:**
```bash
python direct_estimate.py
```

**Mode d'ús:**
1. Escull una opció:
   - `1`: Executar experiments complets
   - `2`: Visualitzar resultats existents
   - `3`: Executar un únic experiment amb configuració base

**Paràmetres configurables:**
- `gamma`: Factor de descompte (valor entre 0 i 1).
- `planning_steps`: Nombre de passos de planificació per cada pas real.
- `epsilon_start`: Valor inicial d'epsilon per a l'exploració.
- `epsilon_end`: Valor mínim d'epsilon.
- `epsilon_decay`: Factor de decaïment d'epsilon.
- `custom_reward`: Funció personalitzada per modificar les recompenses.

### 🔍 Q-Learning (`q_learning.py`)

<div style="background-color: #fff1f0; padding: 10px; border-left: 5px solid #ff4d4f; margin-bottom: 15px;">
Q-Learning és un algorisme d'aprenentatge per reforç que aprèn els valors Q mitjançant interaccions directes amb l'entorn.
</div>

**Execució:**
```bash
python q_learning.py
```

**Mode d'ús:**
1. Escull una opció (escriu el número corresponent):
   - `0`: Només visualitzar resultats d'entrenaments previs
   - `1`: Executar un entrenament estàndard
   - Qualsevol altre valor: Executar tots els experiments

**Paràmetres configurables:**
- `NUM_EPISODES`: Nombre d'episodis a entrenar.
- `GAMMA`: Factor de descompte.
- `LEARNING_RATE`: Taxa d'aprenentatge.
- `EPSILON`: Valor inicial d'epsilon per a l'exploració.
- `EPSILON_DECAY`: Factor de decaïment d'epsilon.
- `EPSILON_MIN`: Valor mínim d'epsilon.
- `LEARNING_RATE_DECAY`: Factor de decaïment de la taxa d'aprenentatge.

### 🚀 REINFORCE (`reinforce.py`)

<div style="background-color: #f0f5ff; padding: 10px; border-left: 5px solid #597ef7; margin-bottom: 15px;">
REINFORCE és un algorisme de gradient de política que optimitza directament la política sense calcular explícitament els valors Q.
</div>

**Execució:**
```bash
python reinforce.py
```

**Mode d'ús:**
1. Escull una opció:
   - `1`: Executar experiments complets (3000 episodis)
   - `2`: Visualitzar resultats existents
   - `3`: Executar un únic experiment amb configuració base (3000 episodis)

**Paràmetres configurables:**
- `gamma`: Factor de descompte (valor entre 0 i 1).
- `learning_rate`: Taxa d'aprenentatge per a l'optimitzador.
- `num_episodes`: Nombre d'episodis a entrenar.
- `custom_reward`: Funció personalitzada per modificar les recompenses.

## 🎯 Personalització de les Recompenses

Tots els algorismes permeten personalitzar la funció de recompensa. S'inclouen tres opcions predefinides:

1. `default_reward`: Utilitza la recompensa original sense modificacions.
2. `step_penalty_reward`: Afegeix una petita penalització per cada pas per fomentar camins més curts.
3. `cliff_avoidance_reward`: Augmenta la penalització per caure pel precipici.

## 📈 Visualitzacions

Tots els algorismes inclouen visualitzacions útils:

- **Funció de valor**: Mostra el valor esperat de cada estat.
- **Política**: Mostra la política òptima apresa.
- **Progressió d'entrenament**: Gràfiques de recompensa i passos per episodi.
- **Comparacions de paràmetres**: Gràfiques comparatives dels diferents experiments.

## 📚 Referències

- [Gymnasium - CliffWalking](https://gymnasium.farama.org/environments/toy_text/cliff_walking/)
- [Grau en Sistemes d'Informació i Dades (UPC)](https://sites.google.com/upc.edu/grau-sid)

## 👨‍💻 Autors

- [enric.segarra@estudiantat.upc.edu](mailto:enric.segarra@estudiantat.upc.edu)
- [marc.font.cabarrocas@estudiantat.upc.edu](mailto:marc.font.cabarrocas@estudiantat.upc.edu)
- [pablo.calomardo@estudiantat.upc.edu](mailto:pablo.calomardo@estudiantat.upc.edu)