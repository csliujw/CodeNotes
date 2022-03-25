package com.itheima.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.servlet.DispatcherServlet;
import org.springframework.web.servlet.HandlerAdapter;

import java.lang.reflect.Field;
import java.util.List;

@SuppressWarnings("all")
public class DispatcherServletUtil {

    private static final Logger log = LoggerFactory.getLogger(DispatcherServletUtil.class);

    public static void showHandlerAdapter(DispatcherServlet servlet) {
        try {
            Field handlerAdapters = DispatcherServlet.class.getDeclaredField("handlerAdapters");
            handlerAdapters.setAccessible(true);
            List<HandlerAdapter> list = (List<HandlerAdapter>) handlerAdapters.get(servlet);
            System.out.println(">>>>>>>>>>>> handlerAdapters <<<<<<<<<<<<");
            for (int i = 0; i < list.size(); i++) {
                HandlerAdapter handlerAdapter = list.get(i);
                System.out.println((i + 1) + ". " + handlerAdapter.getClass().getSimpleName());
            }
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }
}
