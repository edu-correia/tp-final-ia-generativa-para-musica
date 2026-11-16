import muspy

# Carrega e baixa o dataset EMOPIA automaticamente
emopia = muspy.EMOPIADataset(
    "dataset/EMOPIA/",
    download_and_extract=True
)

# Prepara os dados para leitura
emopia.convert()

# Exibe a quantidade de exemplos
print("Quantidade de músicas:", len(emopia))

# Acessa a primeira música
musica = emopia[0]

# Exibe a anotação emocional
print("\nAnotação da primeira música:")
print(musica.annotations[0].annotation)

# Exibe informações musicais
print("\nInformações musicais:")
print("Número de notas:", len(musica.notes))
print("Resolução:", musica.resolution)
