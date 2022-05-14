import argparse
import odometry as vo
import vofromscratch as vofs

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Module selection: 1-vo with opencv or 2-vo from scratch')
    parser.add_argument("mode",  type=int,
                        help="0: vo with opencv; 1: vo from scratch")
    frame_path = 'video_train'
    args = parser.parse_args()

    if (args.mode == 0):
        odometryc = vo.Odometry(frame_path)
    else:
        odometryc = vofs.Odometry(frame_path)

    path = odometryc.run()

