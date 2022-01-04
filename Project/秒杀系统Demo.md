# 秒杀系统

## 场景

- 电商抢购商品
- 火车票抢座 12306

## 为什么需要

> 为什么需要把秒杀这个业务场景单独做成一个系统？

系统并发量小的话，完全不用担心并发请求则不用考虑秒杀系统。如果系统在某些情况下需要考虑高并发访问和下单，那么就需要一套完整的流程保护措施，来保证系统在用户流量高峰期不会挂掉。

- <span style="color:red">严格禁止超卖</span>
- 防止黑产：某几个人恶意刷单，把原本只能一人一份的全部收入囊中
- 保证用户体验：高并发下别出现网页打不开，无法支付，进不去购物车等情况。

## 保护措施

- 乐观锁防止超卖 --- 核心基础
- 令牌桶限流
- Redis 缓存
- 消息队列异步处理消息

# 实现

## 防止超卖

网页卡住，大家没参与到活动，最多是用户吐槽，但是超买了，用户拿不到东西，问题就大了（投诉、赔偿等）

> 数据库表

```sql
-- ----------------------------
-- Table structure for stock
-- ----------------------------
DROP TABLE IF EXISTS `stock`;
CREATE TABLE `stock`  (
  `id` int(0) NOT NULL AUTO_INCREMENT,
  `name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '名称',
  `count` int(0) NOT NULL COMMENT '库存',
  `sale` int(0) NOT NULL COMMENT '已售',
  `version` int(0) NOT NULL COMMENT '乐观锁，版本号',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for order
-- ----------------------------
DROP TABLE IF EXISTS `stock_order`;
CREATE TABLE `stock_order`  (
  `id` int(0) NOT NULL AUTO_INCREMENT,
  `sid` int(0) NULL DEFAULT NULL COMMENT '库存id、商品id',
  `name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '商品名称',
  `create_time` datetime(0) NULL DEFAULT NULL COMMENT '创建时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 826 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of order
-- ----------------------------
INSERT INTO `order` VALUES (826, 1, '鸿星尔克球鞋', '2021-08-07 15:13:36');
INSERT INTO `order` VALUES (827, 1, '鸿星尔克球鞋', '2021-08-07 15:13:36');
INSERT INTO `order` VALUES (828, 1, '鸿星尔克球鞋', '2021-08-07 15:13:36');

```

