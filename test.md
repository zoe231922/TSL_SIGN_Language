# Baasid 後端面試專案

Spring Boot + PostgreSQL + JWT 商品管理 API

## 技術棧

- **Language**: Java 21
- **Framework**: Spring Boot 3.4
- **Database**: PostgreSQL
- **Security**: JWT (jjwt)
- **API Documentation**: Swagger / OpenAPI 3 (springdoc)
- **Build Tool**: Gradle

## 專案結構

```
src/main/java/com/baasid/
├── BaasidApplication.java          # 啟動入口
├── config/
│   ├── SecurityConfig.java         # Spring Security 設定
│   ├── OpenApiConfig.java          # Swagger 設定
│   └── DataInitializer.java        # 測試資料初始化（dev profile）
├── controller/
│   ├── AuthController.java         # 認證 API
│   └── GoodsController.java        # 商品 CRUD API
├── dto/
│   ├── LoginRequest.java           # 登入請求
│   ├── LoginResponse.java          # 登入回應
│   ├── GoodsRequest.java           # 商品請求
│   ├── GoodsResponse.java          # 商品回應
│   └── ErrorResponse.java          # 錯誤回應
├── entity/
│   ├── SystemUser.java             # 使用者 Entity
│   └── Goods.java                  # 商品 Entity
├── exception/
│   ├── GlobalExceptionHandler.java # 全域例外處理
│   ├── UnauthorizedException.java
│   └── ResourceNotFoundException.java
├── repository/
│   ├── SystemUserRepository.java
│   └── GoodsRepository.java
├── security/
│   ├── JwtUtil.java                # JWT 工具類別
│   └── JwtAuthenticationFilter.java # JWT 過濾器
└── service/
    ├── AuthService.java            # 認證服務
    └── GoodsService.java           # 商品服務
```

## 環境需求

- Java 21+
- PostgreSQL 13+
- Gradle 8+ (或使用內建 Gradle Wrapper)

## 啟動方式

### 1. 建立資料庫

```bash
# 連線 PostgreSQL
psql -U postgres

# 建立資料庫
CREATE DATABASE baasid;
\q
```

### 2. 方式 A：使用 SQL 腳本初始化資料

```bash
psql -U postgres -d baasid -f sql/init.sql
```

然後啟動專案：

```bash
./gradlew bootRun
```

### 2. 方式 B：自動初始化（使用 dev profile）

使用 dev profile 啟動，會自動建立表格和測試資料：

```bash
./gradlew bootRun --args='--spring.profiles.active=dev'
```

### 3. 確認啟動成功

啟動後瀏覽 Swagger UI：

```
http://localhost:8080/swagger-ui.html
```

## 測試帳號

| 帳號   | 密碼         | 名稱     |
|--------|-------------|----------|
| admin  | password123 | 管理員   |
| user01 | password123 | 測試用戶 |

## API 使用範例

### 1. 登入取得 Token

```bash
curl -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"account":"admin","password":"password123"}' \
  -i
```

從 Response Header 的 `Authorization` 欄位取得 `Bearer <token>`。

### 2. 新增商品

```bash
curl -X POST http://localhost:8080/goods/add \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"goods_name":"新商品"}'
```

### 3. 取得所有商品

```bash
curl -X GET http://localhost:8080/goods \
  -H "Authorization: Bearer <token>"
```

### 4. 取得指定商品

```bash
curl -X GET http://localhost:8080/goods/{id} \
  -H "Authorization: Bearer <token>"
```

### 5. 更新商品

```bash
curl -X PUT http://localhost:8080/goods/{id} \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"goods_name":"更新後的商品名稱"}'
```

### 6. 刪除商品

```bash
curl -X DELETE http://localhost:8080/goods/{id} \
  -H "Authorization: Bearer <token>"
```

## 資料庫設定

預設連線設定（可在 `application.yml` 修改）：

| 設定     | 值                                 |
|----------|-------------------------------------|
| URL      | jdbc:postgresql://localhost:5432/baasid |
| Username | postgres                            |
| Password | postgres                            |
| Port     | 5432                                |
