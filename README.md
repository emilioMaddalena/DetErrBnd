# DetErrBnd

This code is the supplementary material for the paper

Deterministic error bounds for kernel-based learning techniques under bounded noise

```
@article{maddalena2021embedded,
  title={Embedded PWM predictive control of DC-DC power converters via piecewise-affine neural networks},
  author={E. T. Maddalena, P. Scharnhorst and C. N. Jones},
  journal={Automatica (accepted)},
  volume={-},
  pages={1--18},
  year={2021}
}
```

## Description :books:

We derive deterministic error-bounds for two non-parametric kernel models: kernel ridge regression (KRR) and support vector regression (SVR). Our setting is that of bounded measurement noise. Our expressions only involve solving a single box-constrained quadratic program off-line, but otherwise are given in closed-form.

## Dependencies  :building_construction:

- YALMIP (https://yalmip.github.io/)
- Gurobi (https://www.gurobi.com/)

![alt text](https://github.com/emilioMaddalena/KPC/blob/dev/fig/pred.png)