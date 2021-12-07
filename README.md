# OpenStereo

This is an awesome open source toolbox for stereo matching.

## Supported Methods:

- [x] BM
- [x] SGM(T-PAMI'07)
- [x] GCNet(ICCV'17)
- [x] PSMNet(CVPR'18)
- [x] StereoNet(ECCV'18)
- [x] CFPNet(ICIEA'19)
- [x] ECA(AAAI'18)
- [x] HSMNet(CVPR'19)
- [x] GwcNet(CVPR'19)
- [x] AnyNet(ICRA'19)
- [ ] STTR(ICCV'21)

## KITTI2015 Validation Results

|  Models   | bad-1(％) | bad-3(％) | bad-5(％) | EPE  | RMSE |
| :-------: | :-------: | :-------: | :-------: | :--: | :--: |
|    BM     |           |           |           |      |      |
|    SGM    |           |           |           |      |      |
|   GCNet   |           |           |           |      |      |
|  PSMNet   |           |           |           |      |      |
| StereoNet |           |           |           |      |      |
|  CFPNet   |           |           |           |      |      |
|    ECA    |           |           |           |      |      |
|  HSMNet   |           |           |           |      |      |
|  GwcNet   |           |           |           |      |      |
|  AnyNet   |           |           |           |      |      |
|   STTR    |           |           |           |      |      |

## PlantStereo Validation Results

|  Models   | bad-1(％) | bad-3(％) | bad-5(％) | EPE  | RMSE |
| :-------: | :-------: | :-------: | :-------: | :--: | :--: |
|    BM     |           |           |           |      |      |
|    SGM    |           |           |           |      |      |
|   GCNet   |           |           |           |      |      |
|  PSMNet   |           |           |           |      |      |
| StereoNet |           |           |           |      |      |
|  CFPNet   |           |           |           |      |      |
|    ECA    |           |           |           |      |      |
|  HSMNet   |           |           |           |      |      |
|  GwcNet   |           |           |           |      |      |
|  AnyNet   |           |           |           |      |      |
|   STTR    |           |           |           |      |      |

## PlantStereo Test Results

|  Models   | bad-1(％) | bad-3(％) | bad-5(％) | EPE  | RMSE |
| :-------: | :-------: | :-------: | :-------: | :--: | :--: |
|    BM     |           |           |           |      |      |
|    SGM    |           |           |           |      |      |
|   GCNet   |           |           |           |      |      |
|  PSMNet   |           |           |           |      |      |
| StereoNet |           |           |           |      |      |
|  CFPNet   |           |           |           |      |      |
|    ECA    |           |           |           |      |      |
|  HSMNet   |           |           |           |      |      |
|  GwcNet   |           |           |           |      |      |
|  AnyNet   |           |           |           |      |      |
|   STTR    |           |           |           |      |      |

## References

[1] Hirschmuller, H. (2007). Stereo processing by semiglobal matching and mutual information. *IEEE Transactions on pattern analysis and machine intelligence*, *30*(2), 328-341.

[2] Kendall, A., Martirosyan, H., Dasgupta, S., Henry, P., Kennedy, R., Bachrach, A., & Bry, A. (2017). End-to-end learning of geometry and context for deep stereo regression. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 66-75).

[3] Chang, J. R., & Chen, Y. S. (2018). Pyramid stereo matching network. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 5410-5418).

[4] Khamis, S., Fanello, S., Rhemann, C., Kowdle, A., Valentin, J., & Izadi, S. (2018). Stereonet: Guided hierarchical refinement for real-time edge-aware depth prediction. In *Proceedings of the European Conference on Computer Vision (ECCV)* (pp. 573-590).

[5] Zhu, Z., He, M., Dai, Y., Rao, Z., & Li, B. (2019, June). Multi-scale cross-form pyramid network for stereo matching. In *2019 14th IEEE Conference on Industrial Electronics and Applications (ICIEA)* (pp. 1789-1794). IEEE.

[6] Yu, L., Wang, Y., Wu, Y., & Jia, Y. (2018, April). Deep stereo matching with explicit cost aggregation sub-architecture. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 32, No. 1).

[7] Yang, G., Manela, J., Happold, M., & Ramanan, D. (2019). Hierarchical deep stereo matching on high-resolution images. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 5515-5524).

[8] Guo, X., Yang, K., Yang, W., Wang, X., & Li, H. (2019). Group-wise correlation stereo network. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 3273-3282).

[9] Wang, Y., Lai, Z., Huang, G., Wang, B. H., Van Der Maaten, L., Campbell, M., & Weinberger, K. Q. (2019, May). Anytime stereo image depth estimation on mobile devices. In *2019 International Conference on Robotics and Automation (ICRA)* (pp. 5893-5900). IEEE.

[10] Li, Z., Liu, X., Drenkow, N., Ding, A., Creighton, F. X., Taylor, R. H., & Unberath, M. (2021). Revisiting stereo depth estimation from a sequence-to-sequence perspective with transformers. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 6197-6206).

## Citation

```tex
@misc{OpenStereo,
    title={OpenStereo},
    author={Qingyu Wang and Mingchuan Zhou},
    howpublished = {\url{https://github.com/wangqingyu985/OpenStereo}},
    year={2021}
}
```



## Acknowledgements
This project is mainly based on: [PSMNet](https://github.com/KinglittleQ/PSMNet) thanks to the [author](https://github.com/KinglittleQ).

## Contact

If you have any questions, please do not hesitate to contact us through E-mail or issue, we will reply as soon as possible.

12013027@zju.edu.cn or mczhou@zju.edu.cn