## Concurency Model(in progress of planning)

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
    participant Sync as Синхронизация

    rect rgba(200,230,255,0.5)
        MainThread->>+Logic: frame_start()
        
        %% Параллельные вычисления
        par
            Logic->>+PhysicsPool: enqueue(simulate_physics)
            PhysicsPool-->>Sync: arrive_and_wait(physics)
            and
            Logic->>+MathPool: enqueue(update_matrices)
            MathPool-->>-Logic: math_done
            and
            Logic->>+IOScheduler: co_await load_assets()
            IOScheduler-->>-Logic: assets_loaded
        end
        
        %% Синхронизация
        Sync->>+RenderThread: разрешить рендер
        RenderThread->>+Vulkan: vkQueueSubmit(
        activate Vulkan
            Vulkan->>Vulkan: GPU execution
        deactivate Vulkan
        Vulkan-->>-RenderThread: fences_signaled
        RenderThread-->>-MainThread: frame_complete
    end

    Note over MainThread,Vulkan: Параллелизм CPU-GPU<br>CPU готовит кадр N+1<br>пока GPU обрабатывает кадр N
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
    participant SyncBarrier as Барьер

    MainThread->>+Logic: frame_start()
    
    %% Параллельные вычисления
    par Физика и математика
        Logic->>+PhysicsPool: simulate_physics()
        PhysicsPool->>SyncBarrier: arrive_and_wait()
        and
        Logic->>+MathPool: update_matrices()
        MathPool-->>-Logic: math_done
    end
    
    %% Асинхронные операции IO
    Logic->>+IOScheduler: co_await load_assets()
    IOScheduler-->>-Logic: assets_loaded
    
    %% Подготовка рендера
    RenderThread->>SyncBarrier: arrive_and_wait()
    SyncBarrier-->>RenderThread: proceed
    Logic->>RenderThread: build_command_buffer()
    
    %% Синхронизация Vulkan
    RenderThread->>Vulkan: vkQueueSubmit()
    Vulkan-->>RenderThread: fences_signaled
    
    %% Завершение кадра
    RenderThread-->>MainThread: frame_complete
    
    Note right of SyncBarrier: Двойная синхронизация:<br>1. CPU-CPU барьер<br>2. CPU-GPU fence
```


или 

```mermaid
sequenceDiagram
    participant MainThread
    participant Logic
    participant Physics
    participant Math
    participant IO
    participant Render
    participant Vulkan
    participant Barrier

    MainThread->>Logic: Начало кадра
    
    par Параллельное выполнение
        Logic->>Physics: Запуск физики
        Physics->>Barrier: Готово
        and
        Logic->>Math: Обновление матриц
        Math-->>Logic: Готово
    end
    
    Logic->>IO: Загрузка ассетов
    IO-->>Logic: Ассеты загружены
    
    Render->>Barrier: Ожидание
    Barrier-->>Render: Разрешение
    Logic->>Render: Построение команд
    
    Render->>Vulkan: Отправка команд
    Vulkan-->>Render: Выполнено
    
    Render-->>MainThread: Кадр завершен
    
    Note over Barrier: CPU-CPU синхронизация
    Note over Vulkan: CPU-GPU синхронизация
```

## Render поток
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
Псевдокод да в стиле C++ но верхне уровевый а не конкретная реализация
```


1. **PhysicsPool** (отдельный пул потоков):
```cpp
class PhysicsPool {
public:
    void simulate() {
        tbb::parallel_for(0, bodies.size(), [&](int i) {
            // SIMD-оптимизированные вычисления
            bodies[i].integrate(dt); 
        });
    }
};
```

2. **IOScheduler** (корутины + IO):
   Псевдокод да в стиле C++ но верхне уровевый а не конкретная реализация
```cpp
coro::task<Texture*> load_texture(string path) {
    auto data = co_await io_async_read(path);
    co_return parse_texture(data); // Парсинг в worker-потоке
}
```

3. **RenderThread-Vulkan связка**:
   Псевдокод да в стиле C++ но верхне уровевый а не конкретная реализация
```cpp
void render_thread_func() {
    while (running) {
        wait_for_main_thread_data();
        
        VkCommandBuffer cmd = begin_frame();
        update_uniforms(cmd);
        vkEndCommandBuffer(cmd);
        
        submit_to_vulkan(cmd);
    }
}
```

### Критические пути синхронизации:

1. **Physics → Render**: Vulkan fences + барьер только для CPU синхронизации
   Псевдокод да в стиле C++ но верхне уровевый а не конкретная реализация
```cpp
// Общий барьер для 2 участников (поток физики и рендер-поток)
inline std::barrier physics_render_sync{2};

// В потоке физики:
void physics_thread() {
    simulate_physics();
    physics_render_sync.arrive_and_wait(); // 1/2
}

// В рендер-потоке:
void render_thread() {
    physics_render_sync.arrive_and_wait(); // 2/2 - разблокирует оба
    build_command_buffer();
}
```

2. **IO → Main Thread**:
   Псевдокод да в стиле C++ но верхне уровевый а не конкретная реализация
```cpp
coro::task<void> asset_loading() {
    // Неблокирующее ожидание
    while (!io_completed) {
        co_await coro::suspend_always{};
    }
}
```

### Распределение потоков
| Пул             | Потоки | Размер очереди | Приоритет |
|-----------------|--------|----------------|-----------|
| Physics         | 2-4    | 8              | High      |
| Math            | 2      | 16             | Normal    |
| IO              | 1-2    | 32             | Low       |
| Render          | 1      | 1 (FIFO)       | Realtime  |


### Реализациия +-:

1. Для физики tbb scalable_allocator аллокатор и  tbb::enumerable_thread_specific

2. Для корутин-ожиданий  **таймауты**:
```cpp
co_await wait_with_timeout(physics_ready, 16ms);
```
и
```cpp
co_await resume_on(some_scheduler);
```

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
    D --> H[Worker Threads: Time Sliced Physics]
    E --> I[Worker Threads: Math]
    F --> J[IO Thread]
    G --> K[Network Thread]
    C --> L[Vulkan API]
    H --> M[Task Stealing]
    I --> M
    J --> N[Async I/O]
    K --> O[Async Network I/O]
```

## Threading model(more details)
```mermaid
graph TD
    subgraph Ядро движка
        A[Main Thread] -->|Frame Sync| B[Task Scheduler]
        B --> C[Worker Pool]
        C --> D[Physics Threads]
        C --> E[Render Thread]
        C --> K[Math Thread]
        C --> F[IO Thread]
	    D <-->|Work Stealing| K
    end

    subgraph Внешние системы
        D -->|Transform Data| G[Vulkan]
        E -->|Cmd Buffers| G
        F -->|Asset Data| H[Resource Cache]
    end

    subgraph Синхронизация
        I[Frame Barrier] --> D
        I --> E
        J[Vulkan Fences] --> G
    end

    %% Стили
    classDef physics fill:#f9f,stroke:#333,stroke-width:2px;
    classDef render fill:#bbf,stroke:#333,stroke-width:2px;
    class D physics;
    class E render;

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
## Memory Pools
```mermaid
graph LR
    A[Main Memory] --> B[Double-Buffered Pools]
    A --> C[Thread-Local Pools]
    A --> D[Lock-Free Queues]
    A --> E[Coroutine Frame Arena]
    A --> F[Vulkan Memory Pools]

    B --> B1[Game State Buffer A]
    B --> B2[Game State Buffer B]
    B --> B3[Render Data Buffer A]
    B --> B4[Render Data Buffer B]

    C --> C1[Physics Cache]
    C --> C2[Math Cache]

    D --> D1[Physics -> Render Queue]
    D --> D2[IO -> Logic Queue]
    D --> D3[Network -> Logic Queue]

    E --> E1[Coroutine Frames: 16-32B]

    F --> F1[Staging Buffer Pool]
    F --> F2[Device-Local VRAM Pool]
    F --> F3[Transient Resource Pool]

    B1 -->|Write| G[Physics]
    B2 -->|Read| H[Render]
    C1 -->|Temp| G
    C2 -->|Temp| I[Math]
    D1 -->|Transfer| H
    D2 -->|Transfer| J[Logic]
    D3 -->|Transfer| J
    F1 -->|Upload| F2
    F2 -->|Aliasing| F3
```

## Memory deps
```mermaid
graph TD
    A[Physics Pool] -->|Writes| B[Transform Buffer: Double-Buffered]
    C[Animation Pool] -->|Reads| B
    D[Render Pool] -->|Reads| B
    E[Math Pool] -->|Temp| F[Math Cache: Thread-Local]
    G[IO Scheduler] -->|Enqueues| H[IO -> Logic Queue]
    I[Network Scheduler] -->|Enqueues| J[Network -> Logic Queue]
    K[Vulkan Renderer] -->|Allocates| L[Staging Buffer Pool]
    K -->|Uses| M[Device-Local VRAM Pool]
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
    A[CPU: Main Thread] -->|Allocate| B[Staging Buffer Pool]
    B -->|vkCmdCopy| C[Device-Local VRAM Pool]
    C -->|Aliasing| D[Transient Resources]
    D -->|Frame N+1| E[Reclaimed]
    A -->|Submit| F[Command Buffer]
    F -->|vkQueueSubmit| G[GPU]
    C -->|Bind| G
```
### Vulkan multi queue
```mermaid
graph LR
    MainThread -->|Submits| G[Graphics Queue]
    MainThread -->|Submits| C[Compute Queue]
    Physics -->|Signal| C[Compute Semaphore]
    C -->|Wait| G
```

### Vulkan multithreading

```mermaid
sequenceDiagram
    participant RenderThread
    participant Worker1
    participant Worker2
    participant Vulkan
    
    RenderThread->>Worker1: vkAllocateCommandBuffer
    Worker1->>Vulkan: vkCmdDraw (Mesh 1)
    RenderThread->>Worker2: vkAllocateCommandBuffer
    Worker2->>Vulkan: vkCmdDraw (Mesh 2)
    RenderThread->>Vulkan: vkQueueSubmit (все буферы)
```

**One `VkQueue` for render thread** (but multiple buffers).
**`VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`**
Sync with **`VkTimelineSemaphore`**

### Hot realod
```mermaid
sequenceDiagram
    AssetCompiler->>ResourceManager: Файл изменён
    ResourceManager->>IOThread: Асинхронная перезагрузка
    IOThread->>RenderThread: Обновить GPU ресурсы
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
    title Оптимизированный Timeline кадра (target: 16.6ms @60Hz)
    section CPU
    Physics (4 threads) : 0ms : 2ms
    Math (2 threads) : 1ms : 1.5ms
    IO (coroutines) : 0.5ms : 3ms
    Render Prep : 2ms : 1ms
    section GPU
    Graphics Queue : 2.5ms : 3.2ms
    Compute Queue : 1ms : 2ms
    section Синхронизация
    CPU Barrier : 2ms : 0.1ms
    GPU Fence : 5.5ms : 0.1ms
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
    I --> J[Input snapshots]
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