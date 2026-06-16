# IHC - Reconhecimento de Gestos (TCC)
## Lista de Tarefas (Tasks)

Este arquivo serve para acompanhar o progresso do desenvolvimento do projeto.

---

### 📊 Progresso Geral
- [x] **Etapa 1:** Criar interface gráfica básica com Tkinter para seleção e exibição de câmera. (Concluído)
- [x] **Etapa 2:** Layout mais "clean" (Concluído)
- [x] **Etapa 3:** Reconhecimento de gestos (Concluído)
- [x] **Etapa 4:** Criar dicionarios de gestos (Concluído)
- [x] **Etapa 5:** Usar dicionário (Concluído)
- [ ] **Etapa 6:** Enviar comandos serial
- [ ] **Etapa 7:** Opção de descrição por dicionario

---

### 📋 Detalhamento das Etapas

#### [x] Etapa 1: Interface Gráfica de Captura
* **Descrição:** Criar uma interface GUI interativa usando `customtkinter` e `opencv` para escolher dispositivos e exibir o feed de vídeo.
* **Funcionalidades implementadas:**
  * Detecção dinâmica de câmeras do sistema.
  * Botão de atualização da lista de dispositivos.
  * Ajustes de imagem: Espelhamento (H/V) e Rotação.
  * Controles de vídeo: Pausar/Retomar e Capturar Foto.
  * Exibição de estatísticas de resolução e FPS em tempo real.
  * Execução em thread paralela para evitar travamento da UI.
* **Status:** Concluído com sucesso.

#### [x] Etapa 2: Layout mais "clean"
* **Descrição:**
  * Lipar funções na tela, deixar somente: "Dispositivo de entrada" com a seleção de câmera
  * Botão (engrenagem) de configuração, mover demais configurações para lá e deixar opção de ativar/desativar informações da camera (fps e resolução)
* **Funcionalidades implementadas:**
  * Sidebar limpa: apenas título, botão ⚙ e seleção de câmera.
  * Janela de configurações (CTkToplevel) com ajustes de imagem, controles de vídeo e toggle de exibição de info.
  * FPS e resolução exibidos na barra inferior, visíveis apenas quando ativado nas configurações.
* **Status:** Concluído com sucesso.

#### [x] Etapa 3: Reconhecimento de gestos
* **Descrição:**
  * Integrar media pipe com opencv para reconhecer gestos e desenhar a posição dos dedos na imagem (visível apenas ao ativar nas configurações)
* **Funcionalidades implementadas:**
  * MediaPipe Tasks API integrado (v0.10.30+) com modelo hand_landmarker.task
  * Detecção de até 2 mãos simultâneas com landmarks em tempo real
  * Reconhecimento de 10+ gestos (FIST, OPEN_HAND, POINTING, PEACE, THUMBS_UP, ROCK, etc.)
  * Toggle de landmarks nas configurações (⚙)
  * Desenho de conexões e pontos sobre o frame da câmera
* **Status:** Concluído com sucesso.

#### [x] Etapa 4: Criar dicionarios de gestos
* **Descrição:**
  * CRUD completo de dicionários de gestos
* **Funcionalidades implementadas:**
  * ComboBox na sidebar para selecionar dicionários
  * Botões Novo/Editar/Excluir
  * Janela de edição com: nome, tempo de captura, lista de gestos com tipo/comando
  * **Captura com contagem regressiva**: botão "🎯 Capturar Gesto" que inicia uma contagem regressiva (conforme `capture_time`) e captura o gesto que está sendo feito na câmera ao final
  * Indicador visual de status da captura (contagem, sucesso ou falha)
  * Validação: exige que todos os gestos tenham sido capturados antes de salvar
  * Lista de referência de comandos PyAutoGUI
  * Validação de nomes duplicados
  * Armazenamento em pasta `dicionarios/` com estrutura de subpastas
* **Status:** Concluído com sucesso.

#### [x] Etapa 5: Usar dicionario
* **Descrição:**
  * Execução de comandos serial e computador baseado em gestos
* **Funcionalidades implementadas:**
  * Seleção de porta serial e baud rate quando há comandos seriais
  * Botão Iniciar/Parar runner
  * Execução de pyautogui para comandos de computador (🖥)
  * Envio serial para comandos serial (📡)
  * Cooldown de 1s entre comandos iguais
  * **Log detalhado** em tempo real na sidebar com indicadores visuais:
    * `[🖥 PC]` para comandos de computador executados
    * `[📡 SERIAL]` para comandos seriais enviados
    * `[👁]` para gestos detectados sem mapeamento
    * `[❌]` para erros de execução
  * Resumo de gestos mapeados ao iniciar o runner
* **Status:** Concluído com sucesso.

#### [ ] Etapa 6: Apenas ideias, não executar ainda
* **Descrição:**
  * Opção de fazer comandos com o rosto (eye tracking e comandos com os olhos)
* **Status:** Pendente.

---
