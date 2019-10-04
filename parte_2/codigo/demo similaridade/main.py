from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence, NILCEmbeddings, \
    ELMoEmbeddings, FastTextEmbeddings, BytePairEmbeddings

import text2vec
from annoy import AnnoyIndex


class DocumentSimilarity:

    def __init__(self):
        # initialize the word embeddings
        self.glove_embedding = WordEmbeddings('pt')
        self.bpe_embedding = BytePairEmbeddings('pt')
        #self.flair_embedding_forward = FlairEmbeddings('pt-forward')
        #self.flair_embedding_backward = FlairEmbeddings('pt-backward')
        #self.elmo_embedding = ELMoEmbeddings('pt')
        #self.fast_text_embedding =  FastTextEmbeddings('skip_s600.bin')

    def get_embeddings(self, sentence):

        # document_embeddings = DocumentPoolEmbeddings(
        #    [self.glove_embedding,  # initialize the document embeddings, mode = mean
        #     self.flair_embedding_backward,
        #     self.flair_embedding_forward])

        # Glove + BPE
        document_embeddings = DocumentPoolEmbeddings(
                [self.glove_embedding,
                 self.bpe_embedding])

        # Nilc fasttext 600 emdedding
        #document_embeddings = DocumentPoolEmbeddings(
        #            [self.fast_text_embedding])

        # Flair
        #document_embeddings = DocumentPoolEmbeddings(
        #    [self.flair_embedding_forward])

        # ElMO
        #document_embeddings = DocumentPoolEmbeddings(
        #    [self.elmo_embedding])

        # create an example sentence
        sentence = Sentence(sentence)

        # embed the sentence with our document embedding
        document_embeddings.embed(sentence)

        # now check out the embedded sentence.
        return sentence.get_embedding()


def use_flair(doc_list):
    doc_sim = DocumentSimilarity()
    result = []
    for doc in doc_list:
        doc = pre_processar(doc)
        sentence_vec = doc_sim.get_embeddings(doc)
        result.append(sentence_vec)
        print('.', end='')

    vector_size = len(result[0])
    print(f"Meu vetor tem {vector_size} dimensoes")

    return vector_size, result


def pre_processar(doc):
    # Temos que fazer uma limpeza no texto para ser classificado melhor
    # Uma coisa importante por exemplo, é retirar do texto a ser processado o rodapé
    # Que tem o nome do ministro. Pq assim tem menos chance de agrupar as decicoes do mesmo juiz

    palavras_que_significam_o_fim = [
        "Publique-se",
        "Publiquem",
    ]

    linhas = doc.splitlines()
    total_len = len(linhas)
    for pos, linha in enumerate(linhas):

        if (total_len < 20) or ((pos/total_len) > 0.5): # Ja passei a metade do texto ou ele eh muito pequeno
            encontrei_parada = False
            for palavra in palavras_que_significam_o_fim:
                if linha.startswith(palavra):
                    encontrei_parada = True
                    break
            if encontrei_parada:
                break
    if pos < total_len-1:
        #print(f"-"*10)
        #print(f"Voltei na linha {pos}")
        #print("\n".join(linhas[pos:]))
        return "\n".join(linhas[:pos])
    else:
        #print(f"Nao achei nada")
        return doc

def read_decisoes():
    result = []
    with open('decisoes.txt', 'r', encoding='utf-8') as f:
        buffer = []
        for line in f:
            if line == "NOVA\n":
                result.append('\n'.join(buffer))
                buffer = []
            else:
                buffer.append(line)

    return result


def use_spacy(doc_list):
    print("Montar o text2vec")
    t2v = text2vec.text2vec(doc_list)
    print("Pegar o vetor de wv")
    result = t2v.tfidf_weighted_wv()
    vector_size = len(result[0])
    print(result)
    return vector_size, result


def montar_annony(vetores):
    tamanho_vec = len(vetores[0])
    t = AnnoyIndex(tamanho_vec, metric='angular')
    print("Vou alimentar o annony")
    for i, vec in enumerate(vetores):
        t.add_item(i, vec)
    print("Vou montar a arvore")
    t.build(10)  # 10 trees
    t.save('test.ann')


def evaluate_tree(doc_list, item, n, vec_size):
    u = AnnoyIndex(vec_size, 'angular')
    u.load('test.ann')  # super fast, will just mmap the file
    similar_indexes = u.get_nns_by_item(item, n)  # will find the n nearest neighbors

    print(similar_indexes)
    with open("saida.txt", 'w', encoding='utf-8') as f:
        for similar in similar_indexes:
            f.write('-'*10 + str(similar) + '-'*10 + '\n')
            f.write(doc_list[similar])


if __name__ == '__main__':
    print("Vou ler as decisoes")
    doc_list = read_decisoes()
    print("Vou Gerar os vetores")
    vec_size, doc_vecs = use_flair(doc_list)
    #vec_size, doc_vecs = use_spacy(doc_list)
    print("Vou montar o ANN")
    montar_annony(doc_vecs)
    print("Vou ver os 20 mais proximos do item 6")
    evaluate_tree(doc_list, 6, 20, vec_size=vec_size)

