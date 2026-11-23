# DINOv2 + RAE a 512×512

Framework experimental para reconstrução, geração condicionada por classe ImageNet e preparação de LoRA. O código oficial está em `vendor/RAE`, fixo no commit `a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041`.

## Colab no VS Code

1. Abre `notebooks/colab.ipynb` e seleciona um runtime GPU através da [extensão oficial Google Colab](https://marketplace.visualstudio.com/items?itemName=google.colab).
2. A célula de clonagem usa o URL GitHub deste projeto e inicializa o submodule do RAE. Se o repositório for privado, autentica o Git no runtime antes de clonar.
3. Executa as células pela ordem apresentada. Mantém o PyTorch e o torchvision do runtime. Para guardar resultados no Drive, define `save_to_drive = True` antes de correr os scripts.

Os comandos seguintes correm a partir da raiz do repositório num runtime GPU. A reconstrução usa a imagem de exemplo oficial por omissão. Podes substituí-la com `--image CAMINHO`.

```bash
python -m rae_experiment.sampling.reconstruct
python -m rae_experiment.sampling.generate --class-id 207 --seed 0
python -m rae_experiment.training.prepare_lora --rank 8
```

## Configurações oficiais

- Reconstrução com `DINOv2-B_512.yaml`. DINOv2-B com quatro registos processa a imagem a 448×448 internamente. Patches de 14 dão uma grelha 32×32 com 768 canais. O decoder ViT-XL `ViTXL_n08_i512` usa patches de 16 para reconstruir 512×512.
- Geração com `DiTDH-XL_DINOv2-B_decXL_AG.yaml`. Usa o mesmo RAE e as estatísticas ImageNet-1k a 512. O DiT-XL `ep400` e o DiT-S `ep20` de autoguidance operam sobre o mesmo latente 768×32×32.
- Cada comando descarrega os checkpoints oficiais de que precisa na primeira execução. Os pesos do RAE e DiT ficam em `vendor/RAE/models/`. Os pesos DINOv2 ficam na cache do Hugging Face.

[RAE oficial](https://github.com/bytetriper/RAE) · [Checkpoints oficiais](https://huggingface.co/nyu-visionx/RAE-collections)

## Módulos

`models` carrega configurações e checkpoints, `representations` carrega o RAE congelado, `sampling` contém as duas CLIs e `training` prepara LoRA nas projeções QKV do DiT através de PEFT. A preparação devolve o RAE congelado e o DiT com adaptadores treináveis para um futuro ciclo de adaptação.

`data`, `latent_space`, `conditioning`, `alignment`, `objectives` e `evaluation` reservam espaço para experiências futuras.

## Desempenho da geração

Num runtime CUDA, inicializa os submodules e instala `requirements.txt`. Os scripts usam os checkpoints oficiais e acrescentam uma linha por execução a `outputs/generation_performance.csv`.

```bash
python -m rae_experiment.sampling.generate_sana --batches 5
python -m rae_experiment.sampling.generate_sana_sprint --batches 5
python -m rae_experiment.sampling.generate_bar --batches 5
```

`--batch-size` define as imagens por batch e `--warmup-batches` define o aquecimento. `--output-dir` guarda apenas as imagens medidas. Os modelos SANA aceitam `--prompt` e `--steps`. O SANA-Sprint usa BF16 por omissão. Na T4, usa `--precision fp16`, como na célula do Colab. O CSV regista a precisão selecionada. O BAR usa `--class-id` e regista as 1024 posições autorregressivas na coluna `steps`.

O carregamento inclui o acesso à cache, eventuais downloads e a transferência para a GPU. O tempo total de geração soma apenas os batches medidos, incluindo descodificação e conversão para imagem. A escrita de PNG fica fora da medição. A média e o desvio-padrão populacional usam o tempo de cada batch dividido pelo batch size. Para observar a latência de imagens individuais, usa `--batch-size 1`. O pico de VRAM é a memória CUDA alocada pelo PyTorch durante a fase medida, incluindo os pesos residentes.

A célula do SANA-Sprint no Colab mede um batch em FP16 para validar a execução na T4. Na A6000, corre o script com `--precision bf16 --batches 5` e usa os resultados dessa GPU para comparar desempenho.
