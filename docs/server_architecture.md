### Server

Деплой в K8s

| Компонент    | Userver(C++ HTTP)/Actix(Rust ws)      | ASIO with io_uring/yojimbo            | In Game Auth       |     |
| ------------ | ------------------------------------- | ------------------------------------- | ------------------ | --- |
| **Логика**   | Аутентификация, чат, API, матчмейкинг | Игровой стейт, физика, предсказание   | Check token via ws |     |
| **Протокол** | HTTP/WebSocket (TCP)                  | UDP (yojimbo) + TCP (ASIO + io_uring) | WebSockets         |     |
| **Сеть**     | Высокоуровневые RPC-запросы           | Low-latency пакеты                    | Low-Latency        |     |
```mermaid
graph TD
    %% Клиентская часть
    A[Игровой Клиент] -->|WebSocket: In-match Аутентификация/Чат Game Register grpc to auth service| B[Userver Server]
    A -->|UDP: Игровой трафик + zstd on-fly| C[Game Server ASIO]
    A -->|WebSocket: Матчмейкинг| B

    %% Auth сервис
    A -->|grpc: /register, /login| J[Auth Service ]
    B -->|ws: /validate| J
    C -->|ws: /validate| J
    R[Valkey] -->|ws: cache| J

    %% Userver (HTTP сервер для профилей и прочего + GeoDNS)
    B -->|REST API| D[(Database PostgreSQL)]
    B -->|Кеширование| E[(Valkey)]
    B -->|gRPC| C
    B -->|Valkey PubSub/шардированные каналы / Valkey Streams| C

    %% Игровой сервер (ASIO/yojimbo)
    C -->|Стейт-синхронизация + zstd on-fly| F[World State Manager]
    C -->|Физика| G[Physics Engine]
    C -->|Предсказание| H[Prediction Service]
    C -->|Сетевые сессии| I[Session Manager]
    F -->|БД для стейта + zstd| M[TimeScaleDB]

    %% Взаимодействие компонентов
    F -->|События| G
    G -->|Обновления| H
    H -->|Коррекции| I
    I -->|Пакеты| A

    %% Внешние сервисы
    B -.->|Логи| L[ELK Stack]
    C -.->|Метрики| K[Prometheus]

    %% Трассировка
    B -.->|gRPC Traces| N[Jaeger]
    C -.->|gRPC Traces| N
    N --> |Хранение| O[Jaeger Storage / Elasticsearch + ScyllaDB отдельный кластер]

```


- **Userver** обрабатывает:
    - Регистрацию/логин (`ws/auth/login`), (`grpc/auth/login`).
    - Создание лобби (`ws/match/create`).
    - Чат через WebSocket.
    - GeoDNS
- **ASIO(with io_uring)/yojimbo**:
    - Принимает UDP-пакеты с инпутами игроков.
    - Рассчитывает физику и отправляет стейт.
 *  **Actix(Rust)** 
	  * Auth Service for validation
### ** Сессии**
Session {
    player_id: UUID
    connection_id: int (или token для UDP)
    last_input_time: timestamp
    last_known_state: WorldStateSnapshot
    reconnection_token: UUID (отдается клиенту при подключении)
    is_active: bool
}
- При первом подключении сервер выдает `reconnection_token` игроку.
- Если игрок дропнулся (разрыв TCP/UDP), сервер НЕ удаляет его сессию сразу.
    - Помечает как `is_active = false`.
    - Ставит таймер `reconnect_timeout` (например, 30 секунд).
- Клиент при повторном запуске отправляет `reconnection_token` через HTTP или WebSocket.
- Сервер восстанавливает сессию:
    - Ставит `is_active = true`.
    - Обновляет `connection_id/адрес`.
    - Отправляет клиенту актуальное состояние мира (`last_known_state`) и дельты.
- Хранить `reconnection_token` в сессии + Valkey.
* Lru cache для  для частых запросов
- Понимать, что UDP-порт клиента при реконнекте может быть новым.
    
- Ловить **KeepAlive/Heartbeats** от клиента (если нет heartbeat долго, считаем дисконнектом).


### **Интеграция между серверами**
gRPC


Client - Server model

Auth
```mermaid
sequenceDiagram
  participant Игрок as Игровой Клиент
  participant Userver as Userver Server
  participant GameServer as Game Server ASIO
  participant Auth as Auth Service
  participant PostgreSQL as Database PostgreSQL
  participant Valkey as Valkey

  %% In-match Authentication Flow
  alt In-match Authentication (после подключения к Game Server)
    Игровой Клиент->>GameServer: UDP Игровой трафик + session_key (Implicit)
    activate GameServer
    GameServer->>Valkey: GET session:session_key (Local Cache Lookup)
    Valkey-->>GameServer: User Session Data or NIL (Local Cache)
    alt Session in Local Cache
      GameServer->>GameServer: Validate game_server_ip from cached Session
      GameServer->>GameServer: Apply user data (roles, etc.) to the player session
      GameServer-->>Игровой Клиент: UDP Игровой трафик
    else Session not in Local Cache
      GameServer->>Auth: WebSocket /validate session_key, game_server_ip
      activate Auth
      Auth->>Valkey: Lookup Session (Potentially Cache)
      Valkey-->>Auth: Session Data or NIL
      alt Session Found in Valkey
        Auth->>Auth: Validate session data (ip, user_id, other checks)
        Auth-->>GameServer: OK + User Data
        GameServer->>Valkey: SET session:session_key {user_id, game_server_ip, user_data} (Local Cache Update)
        Valkey-->>GameServer: OK
        GameServer->>GameServer: Apply user data (roles, etc.) to the player session
        GameServer-->>Игровой Клиент: UDP Игровой трафик
      else Session not Found in Valkey
        Auth->>PostgreSQL: Retrieve user session
        PostgreSQL-->>Auth: User session
        Auth->>Auth: Validate user and session
        Auth-->>GameServer: OK + User Data
        GameServer->>Valkey: SET session:session_key {user_id, game_server_ip, user_data} (Local Cache Update)
        Valkey-->>GameServer: OK
        GameServer->>GameServer: Apply user data (roles, etc.) to the player session
        GameServer-->>Игровой Клиент: UDP Ошибка аутентификации.
      end
      deactivate Auth
    end
    deactivate GameServer
  end

  %% WebSocket /validate (From Userver on initial WS Connect) - Auth Validation
  alt WebSocket Authentication (Initial Connection to Userver)
    Игровой Клиент->>Userver: WebSocket /connect (with token)
    activate Userver
    Userver->>Auth: gRPC /validate token
    activate Auth
    Auth->>PostgreSQL: Lookup User Details
    PostgreSQL-->>Auth: User Details
    Auth-->>Userver: OK + user_id + game_server_ip
    deactivate Auth
    Userver->>PostgreSQL: Получить данные пользователя (например, roles, flags, ограничения) по user_id
    PostgreSQL-->>Userver: Данные пользователя
    Userver->>Valkey: Store session information for that User
    Valkey-->>Userver: OK
    Userver-->>Игровой Клиент: WebSocket {"game_server": "1.2.3.4:7777", "session_key": "xyz"}
    deactivate Userver
  end

  %% Valkey Pub/Sub Update (If applicable, based on previous versions) - User Data Changes
  loop User Data Updates
    PostgreSQL->>Valkey: PUBLISH user_data_update {"user_id": user_id, "user_data": new_user_data} (on user data change)
    note right of Valkey: (Optional) Listener in Game Server updates its session cache.
  end


```


```mermaid
sequenceDiagram
    Клиент->>Userver: WS /match/join (С токеном)
    Userver->>GameServer: gRPC StartMatch()
    GameServer->>Клиент: UDP [Match Ready]
    Клиент->>GameServer: UDP [Input]
    GameServer->>Клиент: UDP [World State]
    Клиент->>Userver: WS [Chat Message]
```

### Scaling
```mermaid
graph TD
    %% Игровой сервер (ASIO/yojimbo)
    A[Клиенты] --> B[Load Balancer HAProxy]
    B -->|HTTP/WS| C[Userver Cluster]
    B -->|UDP| D[Game Server Pool]
    C --> E[Shared Valkey Cluster]
    C --> F[Database Sharding]
    D --> E
    D --> G[Global State Service]
```

### Reconnect logic
```mermaid
sequenceDiagram
    participant Client as Игрок (Клиент)
    participant Userver as Userver Server (HTTP/WebSocket)
    participant GameServer as Game Server (ASIO/yojimbo)
    participant SessionManager as Session Manager

    %% Разрыв соединения
    Client-->>GameServer: UDP соединение обрывается
    GameServer->>SessionManager: Пометить сессию is_active = false, запустить reconnect_timeout (30с)

    %% Игрок перезапускает клиент
    Client->>Userver: WS  /auth/reconnect { reconnection_token }
    Userver->>SessionManager: Проверить токен + Bloom Filter and after Redis cell для защиты от брутфорса

    alt Токен валидный
        SessionManager->>Userver: OK, найти сессию
        Userver->>Client: Ответ 200 OK
        
        %% Переподключение
        Client->>GameServer: Новый UDP-соединение (пакет Hello с токеном или connection_id)
        GameServer->>SessionManager: Обновить connection_id/адрес игрока
        SessionManager->>GameServer: Отметить is_active = true
        
        %% Синхронизация состояния
        GameServer->>Client: Отправить last_known_state + дельты
        Client->>GameServer: Начинает слать input-пакеты
    else Токен невалидный
        SessionManager->>Userver: Ошибка (токен не найден/просрочен)
        Userver->>Client: Ответ  Unauthorized
    end

```

### Reconnect timeout
```mermaid
sequenceDiagram
    participant Client as Игрок (Клиент)
    participant GameServer as Game Server (ASIO/yojimbo)
    participant SessionManager as Session Manager


    %% Потеря соединения
    Client-->>GameServer: UDP соединение обрывается
    GameServer->>SessionManager: Пометить сессию is_active = false, старт reconnect_timeout (30с)

    %% Ожидание реконнекта
    Note over SessionManager: Ждем reconnection_token от клиента...

    %% Время истекло
    SessionManager-->>GameServer: reconnect_timeout истек

    %% Завершение сессии WS
    SessionManager->>SessionManager: Удалить сессию из памяти
    GameServer->>GameServer: Освободить ресурсы под игрока (physics, prediction, network buffers)

    %% Оповещение
    GameServer->>Userver: gRPC NotifyPlayerDisconnected(player_id)
    Userver->>Userver: Опционально обновить статус игрока в БД/Valkey (offline)

    %% Клиент не успел
    Client-->>Userver: WS  /auth/reconnect { reconnection_token }
    Userver->>SessionManager: Проверить токен
    SessionManager-->>Userver: Ошибка (сессия уже удалена)
    Userver->>Client: Ответ 410 Gone (или 401 Unauthorized)

```

### Auth Server

---

### **Архитектура Auth Service**
```mermaid
graph TD
    %% Клиенты и внешние системы
    A[Игровой клиент] -->|HTTP/WS /register /login /reconnect| B[Auth Service]
    C[Userver Server] -->|WS: ValidateToken| B
    D[Game Server] -->|WS: ValidateSession| B

    %% Внутренние компоненты Auth Service
    subgraph Auth Service
        B --> E[API Gateway]
        E --> F[Auth Controller]
        E --> G[Session Controller]
        F --> H[Authentication Module]
        G --> I[Session Manager]
        H --> J[(Valkey: Rate Limiting)]
        I --> K[(Valkey: Sessions)]
        I --> L[(Citus: User Data)]
        H --> L
    end

    %% Внешние зависимости
    K --> M[Valkey Cluster]
    L --> N[PostgreSQL HA]
    J --> M
```

---

### **Компоненты и их ответственность**

#### 1. **API Gateway**
- **Роль**: Единая точка входа для всех запросов.
- **Функции**:
  - Маршрутизация (`/register`, `/login`, `/reconnect`).
  - TLS termination.
  - Базовая валидация запросов (JSON schema, параметры).

#### 2. **Auth Controller**
- **Эндпоинты**:
  - `POST /register` – Регистрация (логин/пароль, OAuth, deviceId).
  - `POST /login` – Аутентификация (возврат JWT + refresh token).
  - `POST /logout` – Инвалидация сессии.
- **Интеграции**:
  - Проверка капчи (если нужно).
  - Верификация email/SMS через внешние сервисы (например, Twilio).
* **Кеширование**: 
  - in-memory LRU cache с TTL 1-5 сек. + Valkey

#### 3. **Session Controller**
- **Эндпоинты**:
  - `POST /reconnect` – Восстановление сессии (через `reconnection_token`).
  - `WS /validate` – Валидация токена в реальном времени (для игровых серверов).
- **Генерация токенов**:
  - **Access Token**: JWT с TTL 5 мин (для WS/gRPC) +  OAuth 2.0 Token Binding.
  - **Refresh Token**: TTL 7 дней (хранится в Valkey).
  - **Reconnection Token**: UUIDv4 + TTL 30 сек (только для UDP-сессий).

#### 4. **Authentication Module**
- **Методы аутентификации**:
  - Логин/пароль (bcrypt/scrypt).
  - OAuth 2.0 (Google, Steam, Xbox Live).
  - DeviceID (для мобильных устройств).
- **Защита**:
  - Rate limiting (Redis Cell).
  - Учет неудачных попыток (блокировка после 5 попыток).

#### 5. **Session Manager**
- **Хранение данных**:
  - **Valkey**:
    - Сессии: `session:<token>` → `{player_id, ip, device, expires_at}`.
    - Индексы: `user:<id>:sessions` → Set[token].
  - **PostgreSQL**:
    - Таблица `users`: `id, email, password_hash, 2fa_secret`.
    - Таблица `devices`: `user_id, device_id, last_login`.
- **Методы**:
  - `CreateSession()` – Генерация токенов, запись в Valkey.
  - `ValidateToken()` – Проверка JWT подписи + актуальности в Valkey.
  - `InvalidateSession()` – Удаление из Valkey (при logout).

---

### **Требования к безопасности**
1. **Шифрование**:
   - Все данные в PostgreSQL(Citus): поля `email`, `device_id` – зашифрованы (AES-GCM).
   - Пароли: хешируются scrypt (N=16384, r=8, p=1).
2. **Токены**:
   - JWT подписываются Ed25519 (асимметричная подпись).
   - Refresh токены – одноразовые (использование инвалидирует предыдущий).
3. **Защита от атак**:
   - **Brute Force**: Rate limiting + блокировка IP через Valkey.
   - **Replay Attacks**: Nonce в JWT (проверка через Valkey).
   - **MITM**: HSTS + Certificate Pinning на клиенте.
4. **Секреты**:
   - Ключи шифрования/подписи хранятся в Vault (или K8s Secrets).
   - Ротация ключей каждые 90 дней.

---

### **Интеграция с другими сервисами**
```mermaid
sequenceDiagram
    participant Client
    participant Userver(Auth)
    participant GameServer
    participant PostgreSQL(Citus)
    participant Valkey

    %% Регистрация
    Client->>Userver(Auth): POST /register {email, password}
    Userver(Auth)->>PostgreSQL(Citus): INSERT user
    Userver(Auth)->>Valkey: SET device_id
    Userver(Auth)-->>Client: 200 OK

    %% Логин
    Client->>Userver(Auth): POST /login {email, password}
    Userver(Auth)->>PostgreSQL(Citus): SELECT user + verify password
    Userver(Auth)->>Valkey: SET session {JWT, refresh_token}
    Userver(Auth)-->>Client: 200 OK + access_token + refresh_token

    %% Валидация токена для игрового подключения
    GameServer->>Userver(Auth): POST /validate {access_token}
    Userver(Auth)->>Valkey: GET session by token
    alt valid
        Userver(Auth)-->>GameServer: 200 OK {player_id, rights}
    else invalid
        Userver(Auth)-->>GameServer: 401 Unauthorized
    end

    %% Реконнект игрока
    Client->>Userver(Auth): POST /reconnect {refresh_token}
    Userver(Auth)->>Valkey: GET session by refresh_token
    alt valid
        Userver(Auth)->>Valkey: UPDATE session + issue new access_token
        Userver(Auth)-->>Client: 200 OK + new access_token
    else invalid
        Userver(Auth)-->>Client: 401 Unauthorized
    end

```

---

### **Масштабируемость**
1. **Горизонтальное масштабирование**:
   - Auth Service: 3+ реплики за балансировщиком.
   - Valkey: Cluster из 6 нод (3 master, 3 replica).
   - PostgreSQL(Citus): Чтение через реплики, запись в master.
1. **Шардирование**:
   - Сессии в Valkey шардируются по `player_id`.
   - Пользователи в PostgreSQL(Citus) – по `email_hash`.
1. **Кеширование**:
   - Частые запросы к `ValidateToken` кешируются в локальной памяти (TTL 1 сек).

---

### **Мониторинг**
1. **Метрики**:
   - `auth_login_attempts` (success/failure).
   - `session_reconnect_time_ms`.
   - `token_validation_latency`.
2. **Логи**:
   - Все попытки входа (с IP/deviceId).
   - Подозрительные события (смена устройства/IP при активной сессии).
3. **Трассировка**:
   - Распределенные трейсы через Jaeger (запросы между Userver ↔ Auth ↔ GameServer).

---

### **Резервное копирование**
1. **PostgreSQL(Citus)**:
   - Ежедневные снепшоты + WAL-логи.
   - Реплика в другом датацентре.
1. **Valkey**:
   - RDB + AOF (append-only file).
   - Копии сессий в S3 каждые 5 минут.

---

###   **Auth server реализация**
- **Язык**: Rust 
- **Библиотеки**:
 - Actix 
  - JWT: `jsonwebtoken` (Rust) / 
  - Valkey: `redis-rs` 
  - gRPC: `tonic` (Rust)
  - LRU cache: `lrumap` (Rust)
- **Инфраструктура**:
  - Развертывание: Kubernetes (StatefulSet для PostgreSQL, Deployment для остального).
  - Ingress: Nginx с TLS 1.3.


```mermaid
classDiagram
    class RedisClient {
        +connection_pool: Pool
        +connect(url: String) -> Result<Connection>
        +execute(cmd: Command) -> Result<Response>
    }

    class ConnectionPool {
        -connections: Vec<Connection>
        +get_connection() -> Connection
        +release_connection(conn: Connection)
    }

    class Command {
        +name: String
        +args: Vec<String>
        +to_redis_cmd() -> RedisCmd
    }

    RedisClient --> ConnectionPool
    RedisClient --> Command
```

---



### Concurency  Model
```mermaid
flowchart TD
    %% Сетевые потоки
    subgraph Network Threads [Сетевые потоки ASIO + Userver]
        A1[UDP сервер ASIO/yojimbo]
        A2[HTTP/WebSocket сервер Userver]
        A3[Auth сервер WS Rust Actix]
    end

    %% Очереди для передачи между потоками
    subgraph Queues [Lock-free Очереди]
        Q1[Lock-free Queue UDP inputs]
        Q2[Lock-free Queue TCP requests]
        Q3[SPSC Queue Outgoing Updates]
    end

    %% Игровая логика и физика
    subgraph Game Logic / Physics Threads [Игровая логика и физика TBB Tasks]
        B[Игровая логика Input Processing]
        D[Физика Physics Engine]
        E[World State Manager]
    end

    %% Управление сессиями
    subgraph Session Management [Менеджеры сессий]
        I1[Session Manager UDP]
        I2[Session Manager HTTP/WS]
    end

    %% Внешние сервисы
    subgraph External Services [Инфраструктура DB / Valkey / Monitoring]
        DB1[Citus / TimescaleDB]
        Valkey[Valkey Cache / PubSub]
        L1[ELK Stack Логирование]
        M1[Prometheus Метрики]
        T1[Jaeger Tracing]
    end

    %% Поток передачи данных
    A1 -->|Игровые инпуты| Q1
    A2 -->|Запросы через Protobuf over HTTP/WS | Q2
    A2 -->|WS запросы | A3

    Q1 -->|Batch-обработка| B
    Q2 -->|Асинхронные обработчики| C[API/Chat/Matchmaking Logic]

    B -->|TBB Tasks| D
    D -->|Delta Updates| E
    E -->|Delta Snapshots| Q3

    A1 -->|Новые соединения / пакеты Hello| I1
    A2 -->|Login / Reconnect| I2

    I1 --> B
    I2 --> B

    Q3 -->|Batching пакетов обратно| A1

    %% Базы данных и кеширование
    C -->|Запросы| DB1
    C -->|Чтение/запись| Valkey
    E -->|Запись стейтов| DB1

    %% Метрики и трейсинг
    C -.->|Логирование событий| L1
    B -.->|Метрики состояния| M1
    D -.->|Метрики физики| M1
    I1 -.->|Трейсы соединений| T1
    I2 -.->|Трейсы авторизаций| T1

    classDef network fill:#c5f0ff,stroke:#333,stroke-width:2px;
    classDef logic fill:#ccffd6,stroke:#333,stroke-width:2px;
    classDef infra fill:#ffd6d6,stroke:#333,stroke-width:2px;
    classDef queue fill:#fff3b0,stroke:#333,stroke-width:2px;

    class A1,A2,A3 network;
    class Q1,Q2,Q3 queue;
    class B,D,E logic;
    class DB1,Valkey,L1,M1,T1 infra;


```

Реплицирование
 - userver:
      replicas: 3
  - game-server:
      replicas: 5 (от 2-10 HPA)
  - Valkey-cluster:
      replicas: 6 (3 masters + 3 replicas)
  - postgres:
      replicas: 2 (master + replica)
  - haproxy:
      replicas: 2
  - jaeger:
      replicas: 1 (agent per node)
  - elk-stack:
      logstash: 1
      elasticsearch: 1
      kibana: 1
  - prometheus:
      replicas: 1
  - grafana:
      replicas: 1
  - global-state-service:
      replicas: 2
  Репликация состояния
  - Репликация стейта через **CRDT** (Conflict-Free Replicated Data Types) или **временные снимки в TimescaleDB**.
- Использовать **Kubernetes StatefulSet** для игровых серверов с persistent volume.

## **HA и АвтоScaling**
- **Userver**:  
    Horizontal Pod Autoscaler (HPA) по метрике WebSocket connections или CPU.
- **Game Server Pool**:  
    Автоскейлинг по количеству активных матчей (через custom metrics API).
- **Auth Service**:  
    HPA по latency запросов на валидацию.
- **Valkey/Citus**:  
    Минимум 3 реплики + репликация + Sentinel/Patroni.

## Namespaces
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: game-backend

```

## Сервисы
| Сервис               | Тип                                 | Примечания                              |
| -------------------- | ----------------------------------- | --------------------------------------- |
| Userver Cluster      | Deployment + Service                | HTTP(S)/WS вход в систему и матчмейкинг |
| Game Server Pool     | Deployment + Service (UDP Headless) | Yojimbo, чистые UDP-сессии              |
| Auth Service (Actix) | Deployment + Service                | Проверка токенов, генерация токенов     |
| Valkey Cluster       | StatefulSet + Headless Service      | Для сессий, кешей                       |
| Citus Cluster        | StatefulSet + LoadBalancer Service  | Хранение профилей, пользователей        |
| HAProxy Ingress      | Deployment + LoadBalancer Service   | Для HTTP/WS (Ingress Gateway)           |
| Prometheus, Grafana  | Helm Charts                         | Метрики, алерты                         |
| Jaeger               | Deployment + Service                | Трейсинг gRPC и WS запросов             |
## Типы сервисов
- **Userver** — `ClusterIP` (внутренний балансировщик).
- **Game Servers** — `Headless Service` для прямого доступа по Pod IP.
- **Auth** — `ClusterIP` (Userver и Game-сервера будут обращаться к нему).
- **HAProxy** — `LoadBalancer`.
- **Valkey и Citus** — `StatefulSet` с PV/PVC для хранения данных.

## **Шардирование**

- **Valkey** — Через Valkey Cluster mode.
- **PostgreSQL** — Citus + PgBouncer.

1. GitHub Actions / GitLab CI:
	- Билд Docker-образов `userver`, `auth-service`, `game-server`.
	- Пуш в Docker Registry (например, ghcr.io, AWS ECR).
	
2. Helm Charts для каждого компонента:
    - `userver/`
    - `auth-service/`
    - `game-server/`
    - общие чарты (`Valkey`, `postgres`, `haproxy`, `prometheus`, `jaeger`).

## **Secrets Management**
- Все секреты (JWT keys, DB пароль) — в Kubernetes Secrets + Vault

### Сеть в k8s
```mermaid
flowchart TD
    Client -->|HTTP/WS| HAProxy
    HAProxy --> UserverCluster
    UserverCluster -->|gRPC| AuthService
    UserverCluster -->|gRPC| GameServerPool
    GameServerPool -->|gRPC| AuthService
    UserverCluster -->|Valkey PubSub| ValkeyCluster
    GameServerPool -->|Valkey| ValkeyCluster
    UserverCluster -->| Citus| PostgreSQLCluster(Citus)
    AuthService -->| Citus| PostgreSQLCluster(Citus)
    UserverCluster -->|Jaeger Tracing| Jaeger
    GameServerPool -->|Jaeger Tracing| Jaeger

```

