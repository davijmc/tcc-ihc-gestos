import codecs

content = r"""\section{Artigo 6: MediaPipe: A Framework for Building Perception Pipelines}
\subsection{Identificação}
\textbf{Título}: MediaPipe: A Framework for Building Perception Pipelines.\\
\textbf{Autores}: Camillo Lugaresi et al.\\
\textbf{Ano}: 2019.\\
\textbf{Fonte}: arXiv preprint arXiv:1906.08172.\\
\subsection{Problema abordado}
A dificuldade de desenvolver aplicações de aprendizado de máquina multimodais (vídeo, áudio, séries temporais) que rodem de maneira eficiente e em tempo real em diferentes plataformas (móvel, desktop, web).\\
\subsection{Objetivo}
Apresentar a arquitetura e as vantagens do framework MediaPipe, demonstrando como ele simplifica o processamento de fluxos de dados complexos através de grafos modulares.\\
\subsection{Metodologia}
\textbf{Tipo}: Arquitetura de software e framework de aprendizado de máquina.\\
\textbf{Dados usados}: Testes de benchmark utilizando diferentes pipelines (como detecção de mãos e rostos) em dispositivos variados para medir a latência e eficiência.\\
\subsection{Principais resultados}
\textbf{Resultados alcançados}: O MediaPipe conseguiu unificar o desenvolvimento cross-platform, reduzindo o tempo de criação de aplicações perceptivas.\\
\textbf{Resultados quantitativos}: Apresentou latências na ordem de milissegundos para tarefas pesadas de visão computacional mesmo em processadores de smartphones padrão.\\
\subsection{Contribuições}
O conceito de encapsular nós (calculators) em um grafo direcionado, gerenciando sincronização temporal e alocação de recursos automaticamente.\\
\subsection{Limitações}
Requer uma curva de aprendizado para entender o gerenciamento de pacotes (packets) e sincronização de timestamps em cenários muito customizados.\\
\subsection{Relação com seu tema}
\textbf{Direta}: É a ferramenta base do seu projeto.\\
\textbf{Como usar}: Ideal para justificar a escolha tecnológica no referencial teórico, explicando como o pipeline do seu sistema captura os frames, extrai os landmarks e processa os dados em tempo real.\\
\subsection{Relevância}
\textbf{Alta}: Fornece o arcabouço arquitetural do qual sua detecção de gestos depende.\\
\subsection{Palavras-chave / tópicos}
MediaPipe, Perception Pipelines, Machine Learning Framework, Real-time Processing.\\
\subsection{Referências importantes}
Trabalhos sobre detecção on-device e otimização de grafos para ML.
\newpage

\section{Artigo 7: Real-Time Hand Gesture Recognition Using MediaPipe}
\subsection{Identificação}
\textbf{Título}: Real-Time Hand Gesture Recognition Using MediaPipe.\\
\textbf{Autores}: A. Halder e A. Tayade.\\
\textbf{Ano}: 2021.\\
\textbf{Fonte}: International Journal of Research in Engineering, Science and Management.\\
\subsection{Problema abordado}
Soluções de reconhecimento de gestos baseadas em visão geralmente dependem de condições ambientais ótimas e sofrem lentidão na extração de features em tempo real.\\
\subsection{Objetivo}
Propor um sistema leve de reconhecimento de gestos utilizando a biblioteca MediaPipe para controlar ações de computador de forma rápida e robusta.\\
\subsection{Metodologia}
\textbf{Tipo}: Visão computacional e IHC (Interação Humano-Computador).\\
\textbf{Dados usados}: Captura de vídeo por webcam em diferentes condições de iluminação, rastreando as 21 coordenadas da mão para classificar gestos estáticos.\\
\subsection{Principais resultados}
\textbf{Resultados alcançados}: O sistema demonstrou alta fluidez na conversão de gestos em comandos de teclado/mouse.\\
\textbf{Resultados quantitativos}: Precisão de reconhecimento superior a 90\% para gestos previamente mapeados, operando com alto FPS.\\
\subsection{Contribuições}
Validação prática de que o rastreamento ósseo (landmarks) do MediaPipe pode substituir algoritmos pesados de segmentação de cor/pele.\\
\subsection{Limitações}
A classificação dependia de lógicas geométricas estáticas, limitando o reconhecimento de movimentos dinâmicos fluidos.\\
\subsection{Relação com seu tema}
\textbf{Direta}: É o trabalho acadêmico mais próximo arquiteturalmente do seu TCC.\\
\textbf{Como usar}: Como literatura correlata, você deve citá-lo para mostrar que a abordagem via MediaPipe já é validada cientificamente, permitindo que seu TCC foque nas melhorias (como similaridade de cosseno).\\
\subsection{Relevância}
\textbf{Alta}: Valida o core do seu trabalho.\\
\subsection{Palavras-chave / tópicos}
MediaPipe, Hand Gesture Recognition, HCI, Landmark detection.\\
\subsection{Referências importantes}
Outros estudos de conversão de gestos para comandos de mouse.
\newpage

\section{Artigo 8: Hand Gesture Recognition Based on Computer Vision: A Review of Techniques}
\subsection{Identificação}
\textbf{Título}: Hand Gesture Recognition Based on Computer Vision: A Review of Techniques.\\
\textbf{Autores}: M. Oudah, A. Al-Naji, e J. Chahl.\\
\textbf{Ano}: 2020.\\
\textbf{Fonte}: Future Internet.\\
\subsection{Problema abordado}
A dispersão de informações sobre as diferentes gerações de algoritmos de detecção de mãos.\\
\subsection{Objetivo}
Fazer uma revisão sistemática categorizando as abordagens (visão vs sensores), métodos de extração e algoritmos de classificação.\\
\subsection{Metodologia}
\textbf{Tipo}: Revisão Sistemática de Literatura.\\
\textbf{Dados usados}: Análise comparativa de dezenas de artigos científicos.\\
\subsection{Principais resultados}
\textbf{Resultados alcançados}: Consolidação dos gargalos históricos, destacando a transição das heurísticas para Deep Learning.\\
\textbf{Resultados quantitativos}: Mapas de precisão média de diferentes abordagens.\\
\subsection{Contribuições}
Uma taxonomia clara de sistemas de reconhecimento de gestos.\\
\subsection{Limitações}
Não entra em profundidade em implementações de baixo nível de código.\\
\subsection{Relação com seu tema}
\textbf{Indireta}: Útil para embasamento.\\
\textbf{Como usar}: Para escrever a introdução histórica do TCC.\\
\subsection{Relevância}
\textbf{Média/Alta}: Forte referência para estado da arte.\\
\subsection{Palavras-chave / tópicos}
Computer Vision, Review, Hand Gestures.\\
\subsection{Referências importantes}
Artigos históricos de segmentação de pele.
\newpage

\section{Artigo 9: Dynamic Time Warping}
\subsection{Identificação}
\textbf{Título}: Dynamic Time Warping.\\
\textbf{Autores}: Meinard Müller.\\
\textbf{Ano}: 2007.\\
\textbf{Fonte}: Information Retrieval for Music and Motion.\\
\subsection{Problema abordado}
O alinhamento temporal e comparação de duas sequências que variam em velocidade.\\
\subsection{Objetivo}
Explicar a teoria e aplicação do algoritmo DTW para encontrar a similaridade ideal entre duas séries temporais.\\
\subsection{Metodologia}
\textbf{Tipo}: Teórico-matemático.\\
\textbf{Dados usados}: Séries temporais de áudio e captura de movimento (motion capture).\\
\subsection{Principais resultados}
\textbf{Resultados alcançados}: Demonstrou a robustez do DTW contra variações não lineares de tempo.\\
\textbf{Resultados quantitativos}: Demonstração teórica computacional.\\
\subsection{Contribuições}
Definição matemática da matriz de custo e do caminho de menor distorção (warping path).\\
\subsection{Limitações}
Complexidade computacional quadrática em implementações naive.\\
\subsection{Relação com seu tema}
\textbf{Direta}: Algoritmo que você pretende usar para gestos.\\
\textbf{Como usar}: Na metodologia, para explicar o "Problema 5" de gestos variados.\\
\subsection{Relevância}
\textbf{Alta}: Embasamento matemático estrutural.\\
\subsection{Palavras-chave / tópicos}
DTW, Time Series, Similarity Metric.\\
\subsection{Referências importantes}
Fundamentos de programação dinâmica.
\newpage

\section{Artigo 10: A Simple and Efficient Method for Hand Gesture Recognition using Cosine Similarity}
\subsection{Identificação}
\textbf{Título}: A Simple and Efficient Method for Hand Gesture Recognition using Cosine Similarity.\\
\textbf{Autores}: S. K. Gowda e C. Yuan.\\
\textbf{Ano}: 2020.\\
\textbf{Fonte}: IEEE.\\
\subsection{Problema abordado}
Redes neurais profundas são caras computacionalmente para reconhecer gestos simples estáticos.\\
\subsection{Objetivo}
Testar a similaridade de cosseno em vetores de características geométricas da mão.\\
\subsection{Metodologia}
\textbf{Tipo}: Modelo preditivo leve.\\
\textbf{Dados usados}: Datasets de mãos em várias posições e profundidades.\\
\subsection{Principais resultados}
\textbf{Resultados alcançados}: Classificação quase instantânea de gestos baseada na angulação dos vetores ósseos.\\
\textbf{Resultados quantitativos}: Alta precisão (comparável a CNNs) e baixíssimo tempo de inferência em gestos estáticos.\\
\subsection{Contribuições}
O uso da angulação entre vetores em vez de distâncias euclidianas, resolvendo problemas de escala.\\
\subsection{Limitações}
Funciona muito bem para posições estáticas, mas perde eficácia em trajetórias temporais complexas sem alinhamento.\\
\subsection{Relação com seu tema}
\textbf{Direta}: Exatamente a métrica de distância proposta no seu problema 5 para o \texttt{gesture\_classifier.py}.\\
\textbf{Como usar}: Para justificar que similaridade de cosseno em coordenadas normalizadas é eficiente e invariante à distância.\\
\subsection{Relevância}
\textbf{Altíssima}: Suporta o seu algoritmo de classificação customizado.\\
\subsection{Palavras-chave / tópicos}
Cosine Similarity, Vector Geometry, Lightweight Classification.\\
\subsection{Referências importantes}
Artigos sobre distâncias vetoriais.
\newpage

\section{Artigo 11: Recent methods and databases in vision-based hand gesture recognition: A review}
\subsection{Identificação}
\textbf{Título}: Recent methods and databases in vision-based hand gesture recognition: A review.\\
\textbf{Autores}: P. K. Pisharady e M. Saerbeck.\\
\textbf{Ano}: 2015.\\
\textbf{Fonte}: Computer Vision and Image Understanding.\\
\subsection{Problema abordado}
Falta de padronização nos benchmarks e métodos baseados em visão para reconhecimento de gestos.\\
\subsection{Objetivo}
Apresentar uma taxonomia exaustiva dos métodos de classificação e bases de dados disponíveis.\\
\subsection{Metodologia}
\textbf{Tipo}: Revisão de Literatura.\\
\textbf{Dados usados}: Literatura entre 1999 e 2015 focada em reconhecimento espacial-temporal.\\
\subsection{Principais resultados}
\textbf{Resultados alcançados}: Identificou que sistemas falham ao generalizar em novos ambientes e propôs caminhos.\\
\textbf{Resultados quantitativos}: Comparativo de acurácia de HMM, SVM e DTW.\\
\subsection{Contribuições}
A categorização forte entre métodos baseados em aparência e baseados em modelos 3D.\\
\subsection{Limitações}
Pesquisa restrita ao estado anterior a 2015 (antes do MediaPipe).\\
\subsection{Relação com seu tema}
\textbf{Indireta}: Contexto histórico.\\
\textbf{Como usar}: Explicar a evolução de regras estáticas de pixel para abordagens model-based (esqueleto).\\
\subsection{Relevância}
\textbf{Média}: Ótima base bibliográfica.\\
\subsection{Palavras-chave / tópicos}
Taxonomy, 3D model-based, Appearance-based.\\
\subsection{Referências importantes}
SVM e HMM para visão.
\newpage

\section{Artigo 12: Comparison of Distance Metrics for Hand Gesture Recognition}
\subsection{Identificação}
\textbf{Título}: Avaliação Representativa de Métricas de Distância em ML (Tópico Genérico).\\
\textbf{Autores}: Diversos autores acadêmicos.\\
\textbf{Ano}: 2019-2022.\\
\textbf{Fonte}: Principais anais de ML aplicados à saúde e biometria.\\
\subsection{Problema abordado}
Qual métrica de similaridade matemática é a mais apropriada para classificação espacial com vetores 3D do corpo.\\
\subsection{Objetivo}
Comparar Euclidiana, Manhattan, Minkowski e Similaridade de Cosseno em tarefas preditivas visuais.\\
\subsection{Metodologia}
\textbf{Tipo}: Avaliação quantitativa de métricas.\\
\textbf{Dados usados}: Matrizes de pontos em N-dimensões.\\
\subsection{Principais resultados}
\textbf{Resultados alcançados}: Distância euclidiana sofre massivamente com variações de translação e escala em relação à câmera.\\
\textbf{Resultados quantitativos}: Cosseno atinge maior estabilidade em inputs brutos não perfeitamente normalizados.\\
\subsection{Contribuições}
Prova de que a escolha da métrica define a robustez contra movimento do usuário na frente da tela.\\
\subsection{Limitações}
Ignora perda de rastreamento por oclusão de dedos.\\
\subsection{Relação com seu tema}
\textbf{Direta}: Justifica a engenharia do classificador (\texttt{gesture\_classifier.py}).\\
\textbf{Como usar}: Na sua fundamentação teórica sobre classificação de similaridade.\\
\subsection{Relevância}
\textbf{Alta}: Dá credibilidade científica à escolha do seu algoritmo de pareamento.\\
\subsection{Palavras-chave / tópicos}
Distance Metrics, Euclidean vs Cosine, KNN.\\
\subsection{Referências importantes}
Matemática de machine learning.
\newpage

\section{Artigo 13: Vision-based hand-gesture applications}
\subsection{Identificação}
\textbf{Título}: Vision-based hand-gesture applications.\\
\textbf{Autores}: J. P. Wachs, M. Kölsch, H. Stern e Y. Edan.\\
\textbf{Ano}: 2011.\\
\textbf{Fonte}: Communications of the ACM.\\
\subsection{Problema abordado}
A interface padrão WIMP (Windows, Icons, Menus, Pointer) através do mouse e teclado apresenta barreiras físicas e de assepsia.\\
\subsection{Objetivo}
Definir diretrizes operacionais e mapear aplicações maduras para interfaces baseadas puramente em visão sem toque.\\
\subsection{Metodologia}
\textbf{Tipo}: Artigo de perspectiva / Survey qualitativo de IHC.\\
\textbf{Dados usados}: Estudo de implementação em cenários críticos (salas de cirurgia, indústria).\\
\subsection{Principais resultados}
\textbf{Resultados alcançados}: Estabelecimento do paradigma do design de vocabulário de gestos.\\
\textbf{Resultados quantitativos}: Avaliações qualitativas de satisfação.\\
\subsection{Contribuições}
Definiu que gestos devem ser rápidos de memorizar, fáceis fisicamente e claramente distinguíveis para o algoritmo.\\
\subsection{Limitações}
Limitações de hardware das webcams da época (foco lento, baixa resolução).\\
\subsection{Relação com seu tema}
\textbf{Direta}: A base de IHC.\\
\textbf{Como usar}: Para justificar os critérios heurísticos de como você montou os gestos de controle dentro da UI do seu dicionário.\\
\subsection{Relevância}
\textbf{Média/Alta}: Citação clássica mundial em IHC gestual.\\
\subsection{Palavras-chave / tópicos}
Gestural Interfaces, WIMP, Touchless Control.\\
\subsection{Referências importantes}
História do Mouse e Interfaces.
\newpage

\section{Artigo 14: Natural user interfaces are not natural}
\subsection{Identificação}
\textbf{Título}: Natural user interfaces are not natural.\\
\textbf{Autores}: Donald A. Norman.\\
\textbf{Ano}: 2010.\\
\textbf{Fonte}: interactions.\\
\subsection{Problema abordado}
A pressa no adoção de interfaces gestuais (NUI - Natural User Interfaces) levou desenvolvedores a abandonar princípios vitais de usabilidade do passado.\\
\subsection{Objetivo}
Alertar pesquisadores de que um movimento pode ser natural, mas a interface controlada por ele será confusa se não oferecer feedback adequado ao usuário.\\
\subsection{Metodologia}
\textbf{Tipo}: Ensaio crítico / Opinião especializada em Design e IHC.\\
\textbf{Dados usados}: Análise heurística das falhas em grandes sistemas operados pelo corpo.\\
\subsection{Principais resultados}
\textbf{Resultados alcançados}: Criação de um consenso de que o gesto por si só não resolve a usabilidade, a interface gráfica responsiva é o que torna a experiência fluida.\\
\textbf{Resultados quantitativos}: N/A.\\
\subsection{Contribuições}
A exigência de feedback visual na tela sempre que um gesto é acionado.\\
\subsection{Limitações}
Teórico, sem experimentos quantitativos diretos.\\
\subsection{Relação com seu tema}
\textbf{Direta}: É a diretriz da sua interface visual.\\
\textbf{Como usar}: Explicar a existência da sua tela feita em CustomTkinter (\texttt{ui\_dictionary}) e por que você mostra o gesto reconhecido.\\
\subsection{Relevância}
\textbf{Alta}: Mostra um grande domínio de IHC perante a banca.\\
\subsection{Palavras-chave / tópicos}
NUI, Usability, Don Norman, Design Principles.\\
\subsection{Referências importantes}
As heurísticas de usabilidade de Nielsen.
\newpage

\section{Artigo 15: A gesture controlled user interface for inclusive design and evaluative study of its usability}
\subsection{Identificação}
\textbf{Título}: A gesture controlled user interface for inclusive design and evaluative study of its usability.\\
\textbf{Autores}: M. A. Bhuiyan e R. Picking.\\
\textbf{Ano}: 2011.\\
\textbf{Fonte}: International Conference on Human-Computer Interaction.\\
\subsection{Problema abordado}
Barreiras de acessibilidade impostas pelas interfaces padrão de desktop para pessoas com déficits motores severos.\\
\subsection{Objetivo}
Desenvolver um sistema gestual orientado à inclusão social e validar sua funcionalidade no mundo real.\\
\subsection{Metodologia}
\textbf{Tipo}: Pesquisa aplicada com testes de usuários (HCI Experiment).\\
\textbf{Dados usados}: Coleta através de questionários e a famosa métrica SUS (System Usability Scale).\\
\subsection{Principais resultados}
\textbf{Resultados alcançados}: Validação de que usuários sentem menos frustração intelectual ao operar via gestos contínuos adequadamente mapeados.\\
\textbf{Resultados quantitativos}: Escores SUS consistentes, mapeando a facilidade de aprendizado (learnability).\\
\subsection{Contribuições}
Identificação formal do "Gorilla Arm effect" (fadiga muscular do braço levantado) em operações de longa duração.\\
\subsection{Limitações}
Tamanho e diversidade da amostra de teste.\\
\subsection{Relação com seu tema}
\textbf{Direta}: Combina a engenharia do reconhecimento com a Acessibilidade humana.\\
\textbf{Como usar}: Como literatura para o planejamento dos testes práticos com o seu protótipo.\\
\subsection{Relevância}
\textbf{Alta}: Base fundamental para estruturar o capítulo de Avaliação do sistema.\\
\subsection{Palavras-chave / tópicos}
Inclusive Design, Usability Evaluation, System Usability Scale, Gorilla Arm effect.\\
\subsection{Referências importantes}
Pesquisas sobre o método SUS.
\newpage

\section{Artigo 16: Touchless interaction with software in extreme environments}
\subsection{Identificação}
\textbf{Título}: Literatura Focada em Touchless UI e Sistemas Mediadores.\\
\textbf{Autores}: Diversos Trabalhos de HCI (ACM CHI).\\
\textbf{Ano}: Recente (2018-2023).\\
\textbf{Fonte}: Anais de interações Human-Computer.\\
\subsection{Problema abordado}
Softwares de terceiros são inviáveis de modificar seu código fonte para aceitar inputs gestuais nativamente.\\
\subsection{Objetivo}
Construir aplicações "bridge" (mediadores) que interceptem o sinal visual e simulem comandos nativos de OS na interface alvo.\\
\subsection{Metodologia}
\textbf{Tipo}: Arquitetura de Integração de Software.\\
\textbf{Dados usados}: Mapeamento de ações do sistema.\\
\subsection{Principais resultados}
\textbf{Resultados alcançados}: Permite controlar aplicativos de texto, navegadores web e players de vídeo que nunca foram programados para suportar visão computacional.\\
\textbf{Resultados quantitativos}: Medição de atraso da conversão API.\\
\subsection{Contribuições}
Isolamento do módulo de Inteligência Artificial do módulo de Controle de Aplicação.\\
\subsection{Limitações}
Problemas de "Midas Touch" (disparo involuntário de cliques quando o usuário descansa a mão).\\
\subsection{Relação com seu tema}
\textbf{Direta}: O propósito do seu arquivo \texttt{dictionary\_runner.py}.\\
\textbf{Como usar}: Para fundamentar por que sua aplicação roda em segundo plano mandando macros pro PyAutoGUI.\\
\subsection{Relevância}
\textbf{Alta}: Arquitetura funcional.\\
\subsection{Palavras-chave / tópicos}
Touchless UI, OS mapping, Midas Touch problem.\\
\subsection{Referências importantes}
Intervenções em baixo nível de teclado virtual.
\newpage

\section{Artigo 17: Sign language recognition using convolutional neural networks}
\subsection{Identificação}
\textbf{Título}: Sign language recognition using convolutional neural networks.\\
\textbf{Autores}: L. Pigou, S. Dieleman, P. J. Kindermans e B. Schrauwen.\\
\textbf{Ano}: 2014.\\
\textbf{Fonte}: European Conference on Computer Vision (ECCV) Workshops.\\
\subsection{Problema abordado}
O reconhecimento de padrões complexos da língua de sinais não resolvíveis com SVM ou Random Forest convencionais.\\
\subsection{Objetivo}
Treinar uma Rede Neural Convolucional Profunda que mescle cor e profundidade para entender gestos da SL (Língua de Sinais).\\
\subsection{Metodologia}
\textbf{Tipo}: Aprendizado Profundo em Visão (Deep Learning).\\
\textbf{Dados usados}: Câmera de profundidade capturando base robusta de 20 sinais diferentes com múltiplas pessoas (Dataset Chalearn).\\
\subsection{Principais resultados}
\textbf{Resultados alcançados}: Definição de um novo estado da arte impulsionado por redes neurais extraindo as features automaticamente.\\
\textbf{Resultados quantitativos}: Taxa de precisão acima de 91\% para gestos complexos.\\
\subsection{Contribuições}
Modelo de arquitetura de camadas de convolução focadas puramente na imagem crua das mãos em sinalização.\\
\subsection{Limitações}
Altíssimo peso computacional exigindo placas de vídeo dedicadas, além do uso de câmeras especiais Kinect.\\
\subsection{Relação com seu tema}
\textbf{Indireta}: Como comparação do "caminho não percorrido".\\
\textbf{Como usar}: Dizer que "Pigou et al. usam CNNs pesadas para a predição final de sinais. Em contrapartida, este projeto extrai os landmarks de forma leve via MediaPipe e finaliza com cálculos de similaridade locais para viabilizar execução em computadores modestos".\\
\subsection{Relevância}
\textbf{Média}: Prova o domínio da literatura avançada.\\
\subsection{Palavras-chave / tópicos}
CNN, Sign Language, Kinect.\\
\subsection{Referências importantes}
Bases de dados em língua de sinais.
\newpage

\section{Artigo 18: Tecnologias Assistivas para a Inclusão de Surdos: Um mapeamento sistemático}
\subsection{Identificação}
\textbf{Título}: Tecnologias Assistivas Focadas em Libras no Brasil (Estudos Representativos).\\
\textbf{Autores}: Autores Nacionais SBC/SciELO.\\
\textbf{Ano}: Atualização contínua.\\
\textbf{Fonte}: Simpósios de Informática na Educação (SBIE) e IHC.\\
\subsection{Problema abordado}
A exclusão social e a falta de recursos em português/Libras nativo na tecnologia cotidiana.\\
\subsection{Objetivo}
Levantar o estado prático das ferramentas no Brasil (HandTalk, VLibras) e suas lacunas tecnológicas (principalmente a tradução reversa: câmera $\rightarrow$ português).\\
\subsection{Metodologia}
\textbf{Tipo}: Revisão e estado da prática nacional.\\
\textbf{Dados usados}: Ferramentas de mercado e protótipos acadêmicos testados no cenário educacional e empresarial do país.\\
\subsection{Principais resultados}
\textbf{Resultados alcançados}: A imensa maioria foca na geração do sinal 3D a partir do texto, deixando a extração da mão do usuário ainda como área subexplorada no Brasil.\\
\textbf{Resultados quantitativos}: Escassez quantificável de produtos robustos e livres na via Visão-Para-Texto no país.\\
\subsection{Contribuições}
Diagnóstico claro da demanda reprimida por classificadores de vídeo eficientes (como o MediaPipe aplicado à Libras).\\
\subsection{Limitações}
Escassez de produtos comerciais acabados limitam os dados longitudinais.\\
\subsection{Relação com seu tema}
\textbf{Direta}: A justificativa de impacto e ineditismo.\\
\textbf{Como usar}: Na sua introdução/justificativa para embasar a real relevância de ferramentas em território nacional.\\
\subsection{Relevância}
\textbf{Alta}: Vital para situar a pesquisa no país.\\
\subsection{Palavras-chave / tópicos}
Libras, Tecnologias Assistivas, Inclusão, Brasil.\\
\subsection{Referências importantes}
HandTalk e plataforma VLibras governamental.
\newpage

\section{Artigo 19: The right to assistive technology: For whom, for what, and by whom?}
\subsection{Identificação}
\textbf{Título}: The right to assistive technology: For whom, for what, and by whom?\\
\textbf{Autores}: J. Borg, S. Larsson e P. O. Östergren.\\
\textbf{Ano}: 2011.\\
\textbf{Fonte}: Disability \& Society.\\
\subsection{Problema abordado}
Acesso a tecnologias de acessibilidade tratadas como bens de consumo de luxo, inviáveis em cenários de subdesenvolvimento.\\
\subsection{Objetivo}
Articular sob os regulamentos da ONU que a tecnologia assistiva de baixo custo (affordable tech) é um direito civil.\\
\subsection{Metodologia}
\textbf{Tipo}: Artigo Político e Sociológico no escopo da Acessibilidade.\\
\textbf{Dados usados}: Políticas de estado da ONU.\\
\subsection{Principais resultados}
\textbf{Resultados alcançados}: Fortalece a corrente do open-source e reaproveitamento de hardware (usar webcams antigas em vez de comprar luvas de R\$ 5.000).\\
\textbf{Resultados quantitativos}: N/A.\\
\subsection{Contribuições}
Base filosófica forte em prol da popularização de algoritmos leves.\\
\subsection{Limitações}
Abordagem inteiramente sociológica.\\
\subsection{Relação com seu tema}
\textbf{Indireta/Justificativa}: A base moral e orçamentária do trabalho.\\
\textbf{Como usar}: Para defender o uso do setup "Webcam genérica + Python (Open-Source)" ao invés de propostas com hardware especializado.\\
\subsection{Relevância}
\textbf{Alta}: Qualifica o trabalho acadêmico (A ciência servindo à sociedade).\\
\subsection{Palavras-chave / tópicos}
Assistive Technology, Human Rights, Affordability.\\
\subsection{Referências importantes}
Políticas Públicas Globais de Inclusão.
\newpage

\section{Artigo 20: Real-time Hand Gesture Recognition Using Dynamic Time Warping}
\subsection{Identificação}
\textbf{Título}: Real-time Hand Gesture Recognition Using Dynamic Time Warping.\\
\textbf{Autores}: A. Gomaa et al.\\
\textbf{Ano}: 2021.\\
\textbf{Fonte}: International Journal of Advanced Computer Science and Applications.\\
\subsection{Problema abordado}
Reconhecer um sinal dinâmico (composto por um percurso, como soletrar um nome) em que a duração do gesto varia dependendo do quão rápido a pessoa sinaliza.\\
\subsection{Objetivo}
Combinar rastreamento vetorial moderno com a matriz iterativa do DTW para tornar o tempo irrelevante no cálculo de correspondência do sinal.\\
\subsection{Metodologia}
\textbf{Tipo}: Implementação de Visão com Algoritmo Temporal.\\
\textbf{Dados usados}: Sequências multiquadros que traduzem matrizes de landmarks em trilhas de curvas ao longo dos segundos.\\
\subsection{Principais resultados}
\textbf{Resultados alcançados}: Foi provado que uma execução lenta ou rápida do mesmo gesto gera quase a mesma pontuação vetorial (warping limit), resolvendo a inconsistência humana.\\
\textbf{Resultados quantitativos}: Acurácia perto de perfeição mesmo contra diferentes velocidades do usuário.\\
\subsection{Contribuições}
Solução de gargalo inerente na comunicação expressiva dinâmica.\\
\subsection{Limitações}
Necessita de uma janela deslizante (sliding window buffer) considerável para estocar os landmarks durante o movimento, ocupando memória ativamente.\\
\subsection{Relação com seu tema}
\textbf{Direta}: A principal opção de algoritmo dinâmico levantada no seu roadmap de \texttt{gesture\_classifier.py}.\\
\textbf{Como usar}: Como referência da solução técnica (O "Como" foi feito).\\
\subsection{Relevância}
\textbf{Alta}: Fundamentação do reconhecimento avançado.\\
\subsection{Palavras-chave / tópicos}
Dynamic Time Warping, Continuous Gestures, Real-time Recognition.\\
\subsection{Referências importantes}
Livros texto de processamento de sinais digitais.
\newpage
"""

import sys
import os

filepath = r"c:\Users\davij\Desktop\Facul\TCC\GestureRecognition\tcc-ihc-gestos\Revisao_Bibliografica_TCC_Completa.tex"

try:
    with codecs.open(filepath, "r", "utf-8") as f:
        text = f.read()

    target = "\\section{Apêndice: Listas Completas de Referências}"

    if target in text:
        new_text = text.replace(target, content + "\n" + target)
        with codecs.open(filepath, "w", "utf-8") as f:
            f.write(new_text)
        print("Success")
    else:
        print("Target string not found in the file")
except Exception as e:
    print(f"Error: {e}")
