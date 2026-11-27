# pMF Any-to-Any no RF100 Corrosion Bi3Q3

## Colab T4

Abre **notebooks/pmf_any_to_any_colab.ipynb** com um runtime T4 e disponibiliza o
clone com a extensão e o export YOLO que já temos. Ajusta REPO, DATASET, CACHE
e RUN no notebook. Para conservar a cache e os checkpoints entre sessões,
coloca CACHE e RUN no Drive. O smoke usa microbatch 1 em FP32 na T4 e regista
o pico de memória CUDA alocada. A célula **Treino completo** exige o smoke e arranca apenas quando START_TRAINING está ativo.

A cópia Windows auditada é **C:/DAOP2/daop/datasets/corrosion-bi3q3**.
Usa os papéis originais do export Roboflow v1, com 840 treino, 304 validação
e 105 teste. A pasta local tem 62 imagens adicionais no teste. O carregador
aplica **extended_test_manifest.json** e verifica os checksums do teste
original. Mantém esse manifesto ao copiar o dataset alargado. O teste fica
fora do precompute e da seleção de modelos. A auditoria verificou duplicados
exatos de ficheiro entre splits. Os dados não incluem grupos de origem
suficientes para demonstrar independência por estrutura ou sequência.

## Auditoria da arquitetura e da loss

- A revisão do conversor Diffusers é a mesma do benchmark existente.
  O script **generate_pmf.py** e os pesos originais continuam intactos.
- A extensão mantém o patch embedder RGB, o RoPE RGB e os blocos de
  atenção originais. Acrescenta uma projeção residual RGB com zero inicial,
  embeddings de papel, modalidade e posição, projeções das outras
  modalidades e LoRA de rank 8 nas projeções da atenção. A atenção é global
  entre os tokens ativos. RGB fica no fim da sequência, como exige o RoPE
  do checkpoint. O teste de identidade compara a saída RGB inicial com a
  mesma conversão sem a extensão.
- Caixas YOLO viram coordenadas xyxy normalizadas em slots ordenados no
  espaço, com classe de ausência. DINOv2-S/14, congelado, faz ROI pooling
  de patch tokens no retângulo de cada caixa. Ambos usam os mesmos índices
  de objeto. Os embeddings são contínuos. O limite inicial é 32 objetos
  por imagem e a auditoria falha caso seja ultrapassado. A presença é
  aprendida por cross-entropy. Os slots vazios ficam fora das losses de
  coordenadas e de embeddings.
- O Depth Anything V2 Small fornece profundidade relativa normalizada
  por imagem, a 64×64. RF100 não fornece profundidade real. AbsolReL e RMSE
  avaliam concordância com esse pseudo-alvo. Os valores perto de zero
  são excluídos de AbsRel com limiar fixo de 0,001.
- As tarefas aleatórias escolhem modalidades ausentes, condicionantes e
  alvos. Alvos contínuos partilham t e r. Os inputs de condição e r ficam
  fixos durante o JVP. Geração RGB livre e edição são amostradas explicitamente.
  A edição usa retângulos das boxes e uma região central de fallback.
- O RGB conserva a parametrização pixel x0, a conversão para velocidade,
  o ruído com escala 4, o termo MeanFlow com JVP sem gradiente e a
  normalização adaptativa. **Isto não reproduz integralmente a loss JAX
  oficial.** O checkpoint de inferência reduzido auditado só tem u. Sem
  a cabeça v, a direção instantânea vem da interpolação conhecida. Se a
  conversão carregada fornecer v, a extensão usa v para o JVP RGB e inclui
  a loss auxiliar. O treino RF100 não aplica orientação ImageNet, o alvo
  guiado por CFG nem as losses perceptuais opcionais do treino original.
  O upstream disponibiliza separadamente um checkpoint completo de treino.
  Para uma comparação fiel desse objetivo integral é necessária uma
  conversão do checkpoint completo e validação de paridade das losses.
- O smoke local usa a conversão reduzida existente em .venv, PyTorch CPU,
  RGB e boxes do export e tensores DINO/profundidade sintéticos. O smoke
  Colab usa o checkpoint 512, dados reais e as duas features pré-computadas.
  Este último é o que habilita o treino.

## Validação e retoma

O config inicial usa 50 épocas, microbatch 1, acumulação para batch 8,
validação por época e checkpoint de pesos treináveis/otimizador/RNG a
cada cinco épocas. Cada época tem um JSON em **RUN/epochs** e exemplos RGB
em **RUN/examples**. O treino exige um smoke CUDA com o mesmo config e splits.

FID e KID usam Inception-v3 com as 304 imagens reais de validação e o
mesmo número de imagens RGB livres, com seed fixa. KID é o estimador MMD
polinomial não enviesado sobre a amostra inteira. FDr é **FDr-DINOv2**,
FD em CLS DINOv2 dividido pelo FD da referência real de treino contra
validação, usando uma subamostra de treino fixa com o mesmo tamanho. Os
resultados são específicos deste protocolo pequeno e domínio.

A avaliação de boxes utiliza COCO mAP sobre RGB real condicionado.
A consistência de geração conjunta usa um detector YOLO independente,
com os três nomes na ordem RF100, aplicado às imagens geradas e comparado
com as boxes geradas. Fornece um checkpoint Ultralytics compatível
treinado exclusivamente no split train. Esta métrica exprime concordância
com o detector. DINO e profundidade são avaliados em reconstrução
condicionada por RGB. LPIPS espacial é agregado dentro e fora do retângulo
sobre a saída bruta, antes da cópia de pixels imóveis, para que o score
exterior não seja trivialmente zero. O PNG de edição mostra a saída final.
A latência é medida em geração conjunta de um passo após sincronização
CUDA, incluindo as projeções e o bloco comum, sem carregar checkpoints.

Após concluir os testes e confirmar a memória da T4, pré-computa todo
o treino/validação e arranca o treino explicitamente. Os caminhos abaixo
referem-se à raiz do clone, numa sessão com as dependências instaladas:

    python -m rae_experiment.any_to_any.data --dataset "$DATASET" --cache "$CACHE"
    python -m rae_experiment.any_to_any.train --config configs/pmf_any_to_any.json \
      --dataset "$DATASET" --cache "$CACHE" --output "$RUN" --detector "$DETECTOR" --train

Para retomar, acrescenta **--resume "$RUN/checkpoints/epoch-005.pt"** com
o checkpoint aplicável. Para desativar uma modalidade, retira-a de
**model.modalities** na configuração e usa outro diretório de execução.
Passa os mesmos **--slots** e **--depth-size** ao precompute caso os alteres.
O baseline original continua em **python -m rae_experiment.sampling.generate_pmf**.

Fontes: [pMF oficial](https://github.com/Lyy-iiis/pMF),
[OmniFlow](https://openaccess.thecvf.com/content/CVPR2025/html/Li_OmniFlow_Any-to-Any_Generation_with_Multi-Modal_Rectified_Flows_CVPR_2025_paper.html),
[MODUS](https://proceedings.mlr.press/v306/ye26a.html) e
[RF100 Corrosion Bi3Q3 v1](https://universe.roboflow.com/roboflow-100/corrosion-bi3q3/dataset/1).
