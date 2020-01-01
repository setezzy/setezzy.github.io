---
title: TestNG 框架之 DataProvider
categories: 
  - Testing
tags: 
  - TestNG
  - testing
---

## 参数化测试

允许使用不同的值运行相同的测试

### 通过@DataProvider传递参数

`@DataProvider`注解的方法返回对象数组，也允许使用迭代器，此时测试方法会一个接一个地调用迭代器返回的值。e.g. `Iterator<Object[]>`

如果 DataProvider 和 test 方法不在同一个类中，需要在 test 方法注解中显式指明：

`@Test(dataProvider="", dataProviderClass="xx.class")`

**支持传递一组参数**

下面的例子，DataProvider返回的是一个二维数组，test 方法被执行的次数与 `object[][]` 包含的一维数组的个数一致

```java
public class TestParameterDataProvider {

    @Test(dataProvider = "provideNumbers")
    public void test(int number, int expected) {
        Assert.assertEquals(number + 10, expected);
    }

    @DataProvider(name = "provideNumbers")
    public Object[][] provideData() {

        return new Object[][] { { 10, 20 }, { 100, 110 }, { 200, 210 } };
    }

}
```

执行上述代码，即运行三个测试用例

**支持传递对象参数**

例如 Map 对象，在 `@Peovider`注解的方法内通过 `put()` 函数传入 key, value，在 `@Test`注解的方法内遍历迭代器获取参数

**根据方法名传递参数**

通过反射获取当前测试方法的方法名，作为参数传递给`method`

下面的例子为在 `@DataProvider`注解的方法内通过条件判断，根据方法名传入不同参数

  ```java
import java.lang.reflect.Method;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class TestParameterDataProvider3 {

    @Test(dataProvider = "dataProvider")
    public void test1(int number, int expected) {
        Assert.assertEquals(number, expected);
    }

    @Test(dataProvider = "dataProvider")
    public void test2(String email, String expected) {
        Assert.assertEquals(email, expected);
    }

    @DataProvider(name = "dataProvider")
    public Object[][] provideData(Method method) {

        Object[][] result = null;

        if (method.getName().equals("test1")) {
            result = new Object[][] {
                { 1, 1 }, { 200, 200 }
            };
        } else if (method.getName().equals("test2")) {
            result = new Object[][] {
                { "test@gmail.com", "test@gmail.com" },
                { "test@yahoo.com", "test@yahoo.com" }
            };
        }

        return result;

    }

}
  ```
  
  **ITestCOntext 运行时参数**
  
  