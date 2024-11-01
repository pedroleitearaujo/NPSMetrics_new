import os
import ollama
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

MODEL_NAME = os.getenv('MODEL_NAME')

content = """
    Analise a lista de comentarios abaixo e me de 10 grupos de produtos que os clientes estão falando.

    Péssimo pois não consigo entrar no meu app
    top comecei hoje e bora ver quantos que vô faturar
    "para você q tem comércio, usa essa financeira, motivo se vender por links para ser melhor para vc, eles bloqueia seu saldo por 30 dia, então vc só vai receber no próximo mês. pfvr não abaixe o app obrigado ton pela péssima experiência agora vou ficar com uma divida atrasada pq o app e lixo"
    aplicativo excelente
    Fiz uma vendo tive que esperar 120 dia co Completou hoje me mandar um email dizendo que o dinheiro Dizendo que devolver o não tem comprovante muito enrolado
    Fiz para poder usar e não consegui. Achei péssimo . Entrei em contato e nada. Então melhor foi desinstalar.
    "Pecimo aplicativo, atendimento horrível prazo de recebimento das vendas muito longos eles dizem que vai pagar e pagar e nunca pagam, não recomendo pra ninguém"
    App mt bom de manusear.
    ótimo aplicativo e às máquinas super indico
    Já instalei e desinstalei esse aplicativo umas 3 ou 4 vezes. Visto que não consigo entrar no APP. Só volta depois da reinstalação.
    vou dar uma estrela por que esse APP é muito ruim toda vez q vou entrar nele apresenta falhas e não consigo entrar no APP pode dar uma melhorada nesse app por que tá muito ruim
    top de mais
    "App bem ruim,Pix não cai na hora e nem da pra ver detalhes do extrato"
    Péssimo aplicativo.pra descontar a taxa dele é rápido mais pra liberar o que é da pessoa é uma burocracia. Imagina se precisasse pra uma emergência não pode contar com ele não viu. Deveria melhorar ou então deixa pra tirar o de vocês depois quando nois receber o nosso também. Péssimo dos péssimo Não indico pra ninguém
    "Efetuei uma venda pelo link e só depois percebi o prazo para receber o valor , um mês de prazo! Achei um absurdo essa demora para receber vendas via link!"
    Boa Noite meu aplicativo não quer abrir eu já botei minha senha e nada de abrir
    "Boa noite não estou conseguindo acessar minha conta da Ton pra fazer indicações,quando vou acessar aparece conta inativa."
    "o aplicativo apresenta falhas, não autorizada o cadastro do dispositivo e após feito as instruções do suporte o mesmo continua cm erros, é impossível de utilizar"
    Estou tendi problemas pra acessar
    Ótimo
    Bom
    tp melhor que tá tendo
    "muito bom, e funciona realmente"
    top demais
    Top
    App dá muito erro na hora de efetuar vendas pelo Tap Ton. Tem que melhorar.
    "Aprovação de cadastro e aplicativo muito fácil de usar. Tap ton funciona muito bem. Sugestão: Já que existe a opção minha Loja no app poderiam haver mais opções gerenciais: Estoque, preço de custo(compra) E poderia haver também opção de venda em dinheiro para controle de caixa. Assim toda informação gerencial seria concentrada no mesmo app"
    "Até é que bom pra conta digital,só deveria deixar ele mais otimizado,quando a gente entra no aplicativo é um pouco lento pra carregar,fora isso eu considero um app bom."
    Atendimento percimo vc bloquearam meu dinheiro meu aplicativo e não consigo mas resgatar meu dinheiro pq aplicativo não abri maís
    "Esse App e muito lento,tenho Internet 5 G,mesmo assim,demora muito para abrir para qualquer atividade,a dos concorrentes é instantâneo,Mercado Pago,Get Net e outros. Precisam melhorar muito !!!!!!! Sou Taxista,sofro muito na rua!!!!!!!"
    "Pra mim nunca funcionou a máquina, nunca dava sinal nos lugares que precisava, a agora com o App usando Tap Ton no celular a mesma coisa e mesmo com WiFi, fica informando que não tem conexão! O App lento e todo dia sai do login! Uma chatura e me faz passar é vergonha!"
    "Minha conta está inativa por conta de um pix , já entrei várias vezes em contato disseram qui até dia 27/06 eu receberia um e-mail, e até agora nada assim fica difícil minhas venda tão indo e meu acesso nada ainda"
    "Estou perdendo vendas, 3 vezes tentei passar vendas, e o aplicativo apresenta erro, aparece o número 4, resolvam isso por gentileza."
    Muito boa
    Pedi pra excluir a minha conta e até hoje não excluíram
    N consigo definir minha chave pix
    Criei minha conta e agora tá aparecendo conta inativa. Conseguem me ajudar?
    Porque tem tanta atualização nesse aplicativo? Haja memória... toda vez que entro pede pra atualizar... acredito que seja desnecessário.
    "Fiz abertura da conta tudo certo,mas ao abrir está informando conta inativa."
    Muito bom
    Apesar de eu ainda não ser nenhum micro empresário vcs estão de parabéns de qualquer forma pois vcs são incríveis tomara que um dia eu seja um micro empresário pra poder fazer parte de vcs mais por hora merece sim as 5 estrelas e nota 10 com toda certeza
    "NAO BOTEM DINHEIRO NO TON, O PIX NAO PEGA , TUDO DA EERO. HORRÍVEL"
    Ótima empresa tudo de bom
    nota 1000
    "Muito satisfatório, agilidade,sem contar que recebo no mesmo dia."
    "Tô com dinheiro na canta mas não tô conseguindo acessar ela sempre dá erro.Tó precisamos do dinheiro e não consigo retirar . Agradeço a atenção,mas resolvi o problema, muito obrigada."
    Não consigo criar a conta no meio do processo fica assim conta inativa 🤬
    Não consigo fazer venda pelo tapton faz tudo certinho mais na hora de passar o cartão atrás do celular só dá erro
    "Não consigo acessar minha conta, pois aparece que ela está inativa, mas ainda tenho dinheiro lá, e não consigo usar a maquininha de cartão por causa disso."
    maravilhosa 😄
    Na última atualização o app diz que minha conta está inativa
    ótimo
    "Estou tendo problema com o app, pois a venda fica em processamento e o dinheiro não cai na conta, já entrei em contato com o atendimento várias vezes e até agora nada se resolveu. E não é a primeira vez, pois da última vez consegui estornar a venda, pois ninguém conseguiu resolver o problema."
    muito bom
    !) .
    "Muito boa,fácil de lida,pena que não cai o dinheiro no mesmo dia."
    "góstaría de deíxa claro que este app apresenta 3 falhas que considero de grande ímpórtáncía para quem utilizá este app, 1- o app não salva ás chaves Pix quando enviamos o píx para clientes 2-o app não oferece cartão de débito no ato do contrato, ele deixa para o sístema analizar se e quando devemos receber um cartão de debíto, isso é um absurdo se soubesse antes, não contrátaria. 3-o app é muito lento demora para abrir, orrivel"
    O app é bom porém exite algumas falhas sobre na hora q precisa criar uma chave pix onde muitas vezes causa erro
    muito bom a aplicativo bem
    A melhor que já tivemos
    ótimo dimais
    Muito bom
    muinto boa.recomendo
    Muito bom eu indico
    otimo
    Muito instabilidade todo dia um defeito diferente
    Por que o APP Não tá fazendo estorno Só tou tendo dor de cabeça
    Compressa maquininha e ela veio com defeito de fábrica a bateria não segura pedi uma nova troca mandar um técnico mas ele não responde as mensagens que a gente envia o suporte não dá o suporte que a gente precisa é uma imundice não compre essa maquininha não vale a pena ela vem com defeito de fábrica
    Bom
    top recomendo a todos que precisen de uma maquininha
    "Aplicativo é horrível. Muito lento pra mudar as funções e não adianta ligar para o suporte , eles não sabem resolver."
    muito boa
    "Atendimento perfeito, maquininha perfeita seria 5* se não fosse o fato de ter que fazer vendas para ter direito ao cartão físico, muitas vezes precisei transferir o valor da conta para usar o cartão, tem lugares que não aceitam pix"
    show
    trabalho com a maquina da ton super maravilhosa
    top
    Muito bom
    muito bom
    maravilhoso super indico 😍
    "excelente app , ótimo recomendo"
    gostei de usar
    Completamente insatisfeita com o serviço. O dinheiro das vendas simplesmente some e não há um bom retorno sobre isso. Recomendo para quem quer ter dor de cabeça.
    Excelente aplicação e soluções adequadas e que facilitam o dia a dia para o empreendedor.
    ótimo
    ótimo aplicativo! Simples e completo .
    muito bom mesmo gostei da maquininha!!
    ótimo
    tudo muito bom 😊
    👍
    "Tive um probleminha com meu app , porem um consultor me orientou e graças a Deus agora está tudo ok, pronto pra vender sem parar!"
    Gosto muito desse ton
    Está dando muitos problemas para entrar.
    ótima
    Ótimo app
    maravilhosa essa empresa! só o fato de vender pelo telefone já é uma maravilha
    O pior aplicativo de maquinha q já existiu
    "Gostei do aplicativo. Só está ruim a demora em abrir o aplicativo. Fica carregando e demora muito, as vezes vc quer fazer um pix rápido e demora muito até abrir. Vamos vê se melhora!"
    as vezes demora muito para abrir o aplicativo
    Boa noite quando vou acessar o app aparece conta inativa. Gostaria de saber o porquê?
    muito bom
"""

response = ollama.chat(
    MODEL_NAME,
    messages=[{'role': 'user', 'content': content}],
)

print(response['message']['content'])