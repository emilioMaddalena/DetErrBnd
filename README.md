# DetErrBnd

This code is the supplementary material for the paper

Deterministic error bounds for kernel-based learning techniques under bounded noise

```
@article{maddalena2021embedded,
  title={Embedded PWM predictive control of DC-DC power converters via piecewise-affine neural networks},
  author={E. T. Maddalena, P. Scharnhorst and C. N. Jones},
  journal={Automatica},
  volume={-},
  pages={1--18},
  year={2021}
}
```

## Description :books:

We make use of deterministic, finite-sample, error bounds for kernel models to design robust MPC controllers. Non-parametric kernel machines are used to learn the dynamics `f(x,u)` of discrete-time dynamical systems. Thanks to the bounds, we can build hyper-rectangles aroung the nominal predictions that are guaranteed to contain the ground-truth states. 

## Dependencies  :building_construction:

- YALMIP (https://yalmip.github.io/)
- Gurobi (https://www.gurobi.com/)

![alt text](https://github.com/emilioMaddalena/KPC/blob/dev/fig/pred.png)
