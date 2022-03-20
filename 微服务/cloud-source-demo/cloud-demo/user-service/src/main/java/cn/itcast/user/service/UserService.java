package cn.itcast.user.service;

import cn.itcast.user.pojo.User;
import org.springframework.stereotype.Service;

@Service
public class UserService {

    public User queryById(Long id) {
        return User.of(1L, "柳岩", "上海市航都路18号黑马程序员");
    }
}