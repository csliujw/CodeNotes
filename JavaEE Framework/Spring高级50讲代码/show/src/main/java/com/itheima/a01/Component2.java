package com.itheima.a01;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.ApplicationEvent;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

@Component
public class Component2 {

    private static final Logger log = LoggerFactory.getLogger(Component2.class);

    @EventListener // 监听事件的方法
    public void aaa(UserRegisteredEvent event) {
        log.debug("{}", event);
        // 用户信息可以通过 event 拿到
        Component1 source = (Component1) event.getSource();
        log.debug("我收到事件了，要执行自己的操作了~发送短信");
    }
}
