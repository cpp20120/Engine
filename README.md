## Engine

Currently at arch stage and come implementations(can be changed)

### Concurency model
```mermaid
graph TD
    A[Main Thread] --> B[TBB: Физика/Математика]
    A --> C[Корутины: I/O]
    A --> D[Vulkan: Рендеринг]
    B --> E[Синхронизация]
    C --> E
    D --> E
```

Vulkan resourses need to use mutex and lock_guard s

```mermaid
sequenceDiagram
    participant MainThread as Главный поток
    participant Logic as Логика (корутины)
    participant PhysicsPool as Пул физики
    participant MathPool as Пул математики
    participant IOScheduler as IO Scheduler
    participant RenderThread as Рендер-поток
    participant Vulkan as Vulkan API

    MainThread->>+Logic: frame_start()
    
    %% Параллельные вычисления
    Logic->>+PhysicsPool: enqueue(simulate_physics)
    Logic->>+MathPool: enqueue(update_matrices)
    
    %% Асинхронные операции
    Logic->>+IOScheduler: co_await load_assets()
    
    %% Ожидание завершения
    PhysicsPool-->>-Logic: physics_done
    MathPool-->>-Logic: math_done
    IOScheduler-->>-Logic: assets_loaded
    
    %% Подготовка рендера
    Logic->>+RenderThread: build_command_buffer()
    
    %% Синхронизация
    RenderThread->>+Vulkan: vkQueueSubmit()
    Vulkan-->>-RenderThread: fences_signaled
    
    %% Завершение кадра
    RenderThread-->>-MainThread: frame_complete
```


###  Архитектурная диаграмма
```mermaid
sequenceDiagram
    participant MainThread as Главный поток
    participant Logic as Логика (корутины)
    participant PhysicsPool as Пул физики
    participant MathPool as Пул математики
    participant IOScheduler as IO Scheduler
    participant RenderThread as Рендер-поток
    participant Vulkan as Vulkan API
    participant SyncBarrier as Барьер (Physics-Render)

    MainThread->>+Logic: frame_start()
    
    %% Параллельные вычисления
    Logic->>+PhysicsPool: enqueue(simulate_physics)
    Logic->>+MathPool: enqueue(update_matrices)
    
    %% Асинхронные операции
    Logic->>+IOScheduler: co_await load_assets()
    
    %% Ожидание завершения
    PhysicsPool->>+SyncBarrier: arrive_and_wait()
    MathPool-->>-Logic: math_done
    IOScheduler-->>-Logic: assets_loaded
    
    %% Подготовка рендера (ждет барьер)
    RenderThread->>+SyncBarrier: arrive_and_wait()
    Logic->>+RenderThread: build_command_buffer()
    
    %% Синхронизация Vulkan
    RenderThread->>+Vulkan: vkQueueSubmit()
    Vulkan-->>-RenderThread: fences_signaled
    
    %% Завершение кадра
    RenderThread-->>-MainThread: frame_complete
```


## Render поток
```mermaid
timeline
    title Frame Timeline
    section Physics
    Simulation : 2ms
    section Render
    Command Buffer : 1ms
    section GPU
    Execution : 3ms
```




### Распределение потоков
| Пул             | Потоки | Размер очереди | Приоритет |
|-----------------|--------|----------------|-----------|
| Physics         | 2-4    | 8              | High      |
| Math            | 2      | 16             | Normal    |
| IO              | 1-2    | 32             | Low       |
| Render          | 1      | 1 (FIFO)       | Realtime  |


### Реализациия +-:

1. Для физики tbb scalable_allocator аллокатор:

2. Для корутин-ожиданий  **таймауты**:

3. В рендер-потоке  **triple buffering**:
-  разделение обязанностей потоков

4.  **Double Buffering** для игрового состояния:
- Устранение contenation  между физикой и рендером
## Иерархия параллелизма

```mermaid
graph TD
    A[Main Thread] --> B[Logic Coroutines]
    A --> C[Render Thread]
    B --> D[Physics Pool]
    B --> E[Math Pool]
    B --> F[IO Scheduler]
    B --> G[Network Scheduler]
    D --> H[Worker Threads: Physics]
    E --> I[Worker Threads: Math]
    F --> J[IO Thread]
    G --> K[Network Thread]
    C --> L[Vulkan API]
    H --> M[Task Stealing]
    I --> M
    J --> N[Async I/O]
    K --> O[Async Network I/O]
```

## Архитектура памяти (RAII based)

```mermaid
graph LR
    A[Main Memory] --> B[Double Buffered]
    B --> C[Game State]
    B --> D[Render Data]
    A --> E[Thread Local]
    E --> F[Physics Cache]
    E --> G[Math Temp]
    A --> H[Lock-Free Queues]
    H --> I[Physics->Render]
    H --> J[IO->Logic]
    H --> K[Network->Logic]
    A --> L[Coroutine Frames]
    L --> M[Stackless 16-32B]
```

## Memory dependencies
```mermaid
graph TD
    A[Physics] -->|Writes| B[Transform Buffer]
    C[Animation] -->|Reads| B
    D[Render] -->|Reads| B
```
## Vulkan Integration
```mermaid
graph BT
    A[Vulkan Device] --> B[Main Thread]
    B --> C[Upload Pool]
    C --> D[Staging Buffer]
    D --> E[GPU Memory]
    B --> F[Command Pool]
    F --> G[Primary CB]
    G --> H[Queue Submit]
```

### Vulkan memory managment
```mermaid
graph LR
    Staging -->|VkCmdCopy| DeviceLocal[VRAM]
    DeviceLocal -->|Aliasing| Transient[Short-lived]
    Transient -->|Frame N+1| Reclaimed
```
### Vulkan multi queue
```mermaid
graph LR
    MainThread -->|Submits| G[Graphics Queue]
    MainThread -->|Submits| C[Compute Queue]
    Physics -->|Signal| C[Compute Semaphore]
    C -->|Wait| G
```

### Task graph
Create own based TBB on and inspired by [UE Task graph](https://github.com/EpicGames/UnrealEngine/blob/release/Engine/Source/Runtime/Core/Private/Async/TaskGraph.cpp) 
```mermaid
graph TD
    A[Main Thread] -->|"Frame Start<br>(submit tasks)"| B[Task Graph]
    
    B -->|"Batch Process<br>(priority sort)"| C[HighPriority Queue]
    B -->|"Batch Process<br>(priority sort)"| D[NormalPriority Queue]
    B -->|"Batch Process<br>(priority sort)"| E[LowPriority Queue]
    
    C -->|"Parallel Dispatch"| F[Physics Tasks]
    C -->|"GPU Upload Prep"| G[Render Prep]
    
    D -->|"Gameplay Thread"| H[AI Logic]
    D -->|"Skinning Thread"| I[Animation]
    D -->|"Async I/O"| O[Network Handling]
    
    E -->|"Background Load"| J[Asset Streaming]
    E -->|"Mem. Maint."| K[Defrag Pool]  
    
    F -->|"TBB Work Stealing"| L[Worker Threads]
    G -->|"Vulkan Cmd Buffers"| L
    H -->|"Behavior Trees"| L
    I -->|"Mesh Deform"| L
    O -->|"Packet Proc."| L
    
    J -->|"Coalesced I/O"| M[IO Thread]
    K -->|"Arena Recycle"| N[Memory Arena]
    
  
    
    classDef queue fill:#f9f,stroke:#333;
    class C,D,E queue;
    classDef io fill:#6f9,stroke:#090;
    class M,J io;
    classDef gpu fill:#69f,stroke:#039;
    class G gpu;
 
```


### Task Graph profiling
```cpp
struct TaskNode {
    tbb::task* task;
    std::chrono::microseconds avg_time;
    uint32_t dependencies;
};
```

Also need  **automatic LOD** for tasks 
## Frame run
```mermaid
sequenceDiagram
    participant MT as MainThread
    participant TG as TaskGraph
    participant WT as WorkerThreads
    
    MT->>TG: Создать задачи
    TG->>WT: HighPriority: Физика
    TG->>WT: HighPriority: Рендер-prep
    WT->>TG: Сигнал завершения
    TG->>MT: Все HighPriority готовы
    MT->>Vulkan: vkQueueSubmit
    TG->>WT: NormalPriority: AI
    TG->>WT: LowPriority: Загрузка
```

## Frame run with networking
```mermaid
sequenceDiagram
    participant MT as MainThread
    participant TG as TaskGraph
    participant WT as WorkerThreads
    participant Net as NetworkScheduler
    
    MT->>TG: Создать задачи
    TG->>WT: HighPriority: Физика
    TG->>WT: HighPriority: Рендер-prep
    TG->>Net: LowPriority: Обработка сетевых пакетов
    WT->>TG: Сигнал завершения
    Net-->>TG: Пакеты обработаны
    TG->>MT: Все HighPriority готовы
    MT->>Vulkan: vkQueueSubmit
    TG->>WT: NormalPriority: AI
    TG->>WT: LowPriority: Загрузка ассетов

```


### Frame pacing
Control CPU-GPU drift via vkAcquireNextImageKHR + at render thread sleep_until(expected_frame_time)


## VERSIONS WITH BATCHING
### Parallelism hierarchy
```mermaid
graph TD
    A[Main Thread] --> B[Frame Graph Builder]
    B --> C[Physics Subgraph]
    B --> D[Render Subgraph]
    B --> E[IO Subgraph]
    C --> F[TBB Tasks]
    D --> G[Vulkan Passes]
    E --> H[Batched Coroutines]
    G --> I[Compute Pass]
    G --> J[Graphics Pass]
    H --> K[Texture Batch]
    H --> L[Model Batch]
```

### Task Graph with bathcing
```mermaid
graph TD
    A[Main Thread] -->|Submit| B[Task Scheduler]
    B --> C[HighPriority]
    B --> D[NormalPriority]
    B --> E[LowPriority]
    C --> F["Physics (Batched TBB Tasks)"]
    D --> G["Render Prep (Frame Graph)"]
    E --> H["IO (Batched Coroutines)"]
    F --> I[Worker Threads]
    G --> J[Render Thread]
    H --> K[IO Thread]
```

### Frame Graph
```mermaid
sequenceDiagram
    participant MT as Main Thread
    participant FG as Frame Graph
    participant Phys as Physics
    participant Render as Render Thread
    participant IO as IO Scheduler

    MT->>FG: Build Frame Graph
    FG->>Phys: Add Physics Nodes (Batched)
    FG->>IO: Add IO Nodes (co_await batch)
    FG->>Render: Add Render Passes
    MT->>FG: Compile & Execute
    Phys->>FG: Complete
    IO->>FG: Complete
    Render->>FG: Submit GPU Work
    FG->>MT: Frame Done
```

### Memory Arch
```mermaid
graph LR
    A[GPU Memory] --> B[Frame Graph Resources]
    B --> C[Physics Data]
    B --> D[Render Targets]
    B --> E[Texture Array Batch]
    A --> F[Staging Buffers]
    F --> G[Batched Uploads]
```

## Single frame graph
```mermaid
sequenceDiagram
    participant MainThread as Главный поток
    participant TaskGraph as Граф задач
    participant Physics as Физика [CPU-Bound]
    participant Logic as Логика [CPU-Bound]
    participant Animation as Анимация [CPU-Bound]
    participant IO as IO задачи [IO-Bound]
    participant Network as Сеть задачи [IO-Bound]
    participant FrameGraph as Построение рендера [CPU-Bound]
    participant RenderThread as Рендер-поток [CPU-Bound → GPU Submit]
    participant Vulkan as Vulkan API
    participant GPU as GPU [GPU-Bound]

    MainThread->>MainThread: Сбор ввода / событий [CPU-Bound]
    MainThread->>TaskGraph: Начало кадра - создание задач [CPU-Bound]

    %% Параллельные задачи
    TaskGraph->>+Physics: Симуляция физики
    TaskGraph->>+Logic: Обновление логики игры
    TaskGraph->>+Animation: Обновление анимаций
    TaskGraph->>+IO: Стриминг ассетов / IO
    TaskGraph->>+Network: Получение и отправка сетевых сообщений

    %% Ожидание завершения
    Physics-->>-TaskGraph: Физика готова
    Logic-->>-TaskGraph: Логика готова
    Animation-->>-TaskGraph: Анимация готова
    IO-->>-TaskGraph: IO готово
    Network-->>-TaskGraph: Сеть готова

    TaskGraph->>FrameGraph: Построение рендер-проходов [CPU-Bound]
    FrameGraph->>RenderThread: Построение команд для GPU [CPU-Bound]
    RenderThread->>Vulkan: Отправка команд в GPU [CPU→GPU]
    Vulkan->>GPU: Выполнение команд [GPU-Bound]

    %% Параллельность CPU и GPU
    Note over MainThread IO Networkgin: CPU начинает подготовку следующего кадра параллельно с GPU работой + принимает сетевые сообщения

    GPU-->>RenderThread: Сигнал о завершении GPU работы
    RenderThread->>MainThread: Завершение кадра (present swapchain)


```

## Latency Graph(~)
 Use VK_EXT_extended_dynamic_state(3) for less draw calls
```mermaid
timeline
    title Frame Latency (Target: 16.6ms @60FPS)
    section CPU (12 threads)
    Input/Events       : 0.5ms
    Physics (4 threads): 2.5ms 
    Game Logic         : 1.2ms
    Animation (Skinning): 1.8ms
    IO Tasks          : 0.3ms (async)
    Frame Graph Setup  : 0.8ms
    Command Recording : 1.5ms
    section GPU
    Compute Passes    : 1.2ms
    G-Buffer Pass     : 1.0ms
    Shadows           : 1.5ms
    Lighting          : 2.0ms
    Post-Processing   : 1.8ms
    section Timeline
    Critical Path     : 0.5ms -> 2.5ms -> 1.5ms -> 2.0ms -> 1.8ms (Total: 8.3ms)
    Parallel Work     : Physics+Logic+Animation overlap (max 2.5ms)
```
```mermaid
timeline
    title 144Hz Optimization (6.94ms budget)
    section CPU
    Physics LOD1      : 1.2ms
    Logic LOD         : 0.8ms
    section GPU
    Shadows LOD       : 0.7ms
    Lighting LOD      : 1.5ms
    Total            : 4.2ms (headroom: 2.74ms)
```

## Network
```mermaid
graph TD
    A[Game Loop] --> B[Network Manager]
    B --> C[Send Queue]
    B --> D[Receive Queue]
    C --> E[IO Service Pool]
    D --> F[Message Dispatcher]
    E --> G[WinSock/IOCP]
    E --> H[ASIO]
    F --> I[Game Systems]
    G -->|Event| F
    H -->|Event| F
```

IOCP with coroutines
```mermaid
classDiagram
    class NetworkService {
        +HANDLE iocp
        +vector<thread> workers
        +start()
        +stop()
    }
    
    class SocketConnection {
        +SOCKET socket
        +coroutine send(buffer)
        +coroutine receive()
    }
    
    NetworkService "1" *-- "n" SocketConnection
```

CoroErrorHandling:
Cancel task if error via std::stop_token

### Cross thread communication
**moodycamel::ConcurrentQueue**

### Build Debug

```sh
mkdir -p build/debug
cd build/debug
cmake --preset debug
cmake --build --preset build-debug
```

### Build Release:
```sh
mkdir -p build/release
cd build/release
cmake --preset release
cmake --build --preset build-release
```

### Vcpkg debug build:
```sh
cmake --preset vcpkg-debug
cmake --build --preset build-vcpkg-debug
```

### Vcpkg release  build:
```sh
cmake --preset vcpkg-release
cmake --build --preset build-vcpkg-release
```


### Build with sanitazers:

## Address sanitizer
```sh
cmake --preset debug-sanitize-address
cmake --build --preset build-debug-sanitize-address
```
## Thread sanitizer
```sh
cmake --preset debug-sanitize-thread
cmake --build --preset build-debug-sanitize-thread
```
## Undefined behavior sanitizer
```sh
cmake --preset debug-sanitize-undefined
cmake --build --preset build-debug-sanitize-undefined```
```
(specify sanitizer what you need)

### Testing

## Run all tests (release build)
```sh
ctest --preset test-all
```

## Run tests with address sanitizer
```sh
ctest --preset test-sanitize-address
```

## Run specific test suite
```sh
ctest --preset test-library1
```

## Run docs generation
```sh
cmake --build . --target docs
```