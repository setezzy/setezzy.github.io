---
title: SSM 项目总结
categories: 
  - Java
  - Spring 框架
tags: spring
---


这篇总结下自己在做网上商城项目时遇到的一些bug，方便日后查阅。 
 
## Spring Framework 

### SSM 整合 

```java
org.springframework.beans.factory.BeanCreationException: Error creating bean with name “org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerAdapter” 

```

原因：缺少 jar 包 

解决：添加以下jar包：`jackson-databind`, `jackson-core`, `jackson-annotations` 
 
```java
PageNotFound:1248 - No mapping for GET /shop_war_exploded/user/index 

```

原因：当controller返回一个路径的时候，该路径（/index.jsp）被当作一个请求被dispatcherServlet 拦截，所以抛出异常 

解决：在 spring 配置文件中加入 

```java
<mvc:default-servlet-handler /> 

```

### 欢迎页设置 

将欢迎页设置为其他路径下的jsp：
 
首先 web.xml 中欢迎页设为空，然后对应路径的controller中加入根路径 

```java
<welcomfile></welcomfile> 
@ResquestMapping("/login", "/") 

```
 
## MyBatis
 
### Mysql 登录错误 

连接数据库时出错：

<pre><code>
Lost connection to MySQL server at 'reading initial communication packet', system error: 102
</code></pre>
 

原因：Mysql 配置文件问题
 
解决（进入配置文件后进行相应修改）： 

```java
mysql --help|grep ‘my.cnf’
sudo vi /etc/my.cnf 
sudo chmod 664 /etc/my.cnf 

```


### mybatis-generator 

使用 generator 生成文件，产生了多余的 `UserWithBolbs` 类 

原因：使用了 version 8.0 的 Connector/J 

解决: 在 jdbcConnection 配置里新增：

```java
<jdbcConnection driverClass="${jdbc.driver}"
                connectionURL="${jdbc.url}"
                userId="${jdbc.username}"
                password="${jdbc.password}">
    <!--新增下面这句-->
    <property name="nullCatalogMeansCurrent" value="true" />
</jdbcConnection>

``` 


### SQL 语法错误 

在插入数据库时提示： 

<pre><code>
You have an error in you SQL syntax … near 'insert into order ...' 
</code></pre>

 
原因：order 为 mysql 关键字 

解决：`order`加上反引号，或改掉数据表名称 

此外，自己碰到的报 sql syntax error 的另一个可能就是传入的 record 为空 


### 传参错误 

传入两个参数进行 sql 查询时报错： 

<pre><code>
nested exception is org.apache.ibatis.binding.BindingException: Parameter 'ostate' not found. Available parameters are [arg1, arg0, param1, param2] 
</code><pre>


原因：传入多个参数时要使用 `@Param` 注解，否则参数只能有一个 

解决： 

```java
List<OrderVO> selectByOrderState(@Param("ostate") Integer ostate, @Param("uid") Integer uid); 

```

## Jquery 

### ajax 返回错误
 
ajax 执行成功，但无返回值。用 alert 测试时发现进入 error 方法体内。 

原因：相应的方法忘了写 @ResponseBody 注解 

Ajax 进入 error 内其他几种可能原因： 

- 返回数据类型不是 json 类型 
- 请求域与当前域不一致 

### 动态元素绑定事件无效 

例如下面动态生成标签： 

```java
<div class="actions"> 
<c:if test="${orderVO.ostate eq 1}"> 
<a id="J_cancelOrder" name="J_cancelOrder" class="btn btn-small btn-line-gray" title="取消订单" data-order-id="${orderVO.oid}" >取消订单</a> 
</c:if> 
<c:if test="${orderVO.ostate eq 3}"> 
<a id="J_closeOrder" name="J_closeOrder" class="btn btn-small btn-line-gray" title="关闭订单" data-order-id="${orderVO.oid}" >关闭订单</a> 
</c:if> 
</div> 

```
 
通过下面的语句绑定 click 事件无效： 

```java
$('#J_cancelOrder').on("click", function() {}) 

```

原因：on 虽然支持给动态元素绑定事件，但 on 前面的元素也必须在页面加载前就存在于 dom 中，动态的元素或样式可以放在 on 的第二个参数 

解决：需要绑定父元素 


```java
$('.actions').on("click", '#J_closeOrder', function() {}) 

```

### css 显示问题 

页面不显示内容或样式 

原因：div 找不到从属关系时不会设置样式，所以要注意 css 中样式的从属关系和 jsp 中的从属关系是否一致 
