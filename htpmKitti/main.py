#!/usr/bin/env python3

# import kitti
import model.services as m
# import online

kp = m.kitti_parser()
x = [0., 1.458974, 2.63547244, 0.96564807, 2.21222542, 1.65225034, 0., 0., 1.,
     2.20176468, 2.40070779, 0.1750559,
     0.20347586, 6.54656438]
kp.save_model(x)
