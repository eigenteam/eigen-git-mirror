#ifndef MANDELBROT_H
#define MANDELBROT_H

#include <Eigen/Array>
#include <QtGui/QApplication>
#include <QtGui/QWidget>

#ifdef REAL
typedef REAL real;
#else
typedef float real;
#endif

enum { packetSize = Eigen::ei_packet_traits<real>::size }; // number of reals in a packet
typedef Eigen::Matrix<real, packetSize, 1> packet; // wrap a packet as a vector
typedef Eigen::Matrix<real, 2, 1> vector2; // really just a complex number, but we're here to demo Eigen !

const int iter = 32; // the maximum number of iterations done per pixel. Must be a multiple of 4.

class MandelbrotWidget : public QWidget
{
    Q_OBJECT

    vector2 center;
    real xradius;
    int size;
    unsigned char *buffer;
    QPoint lastpos;

  protected:
    void resizeEvent(QResizeEvent *);
    void paintEvent(QPaintEvent *);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);

  public:
    MandelbrotWidget() : QWidget(), center(real(0),real(0)), xradius(2),
                         size(0), buffer(0)
    {
      setAutoFillBackground(false);
      setWindowTitle(QString("Mandelbrot/Eigen, sizeof(real)=")+QString::number(sizeof(real))
                     +", sizeof(packet)="+QString::number(sizeof(packet)));
    }
    ~MandelbrotWidget() { if(buffer) delete[]buffer; }
};

#endif // MANDELBROT_H