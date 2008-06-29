#ifndef MANDELBROT_H
#define MANDELBROT_H

#include <Eigen/Array>
#include <QtGui/QApplication>
#include <QtGui/QWidget>

class MandelbrotWidget : public QWidget
{
    Q_OBJECT

    Eigen::Vector2f center;
    float xradius;
    int size;
    unsigned char *buffer;
    QPoint lastpos;
    int draft;

  protected:
    void resizeEvent(QResizeEvent *);
    void paintEvent(QPaintEvent *);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    template<typename Real> int render(int max_iter, int resx, int resy);

  public:
    MandelbrotWidget() : QWidget(), center(0,0), xradius(2),
                         size(0), buffer(0), draft(16)
    {
      setAutoFillBackground(false);
    }
    ~MandelbrotWidget() { if(buffer) delete[]buffer; }
};

#endif // MANDELBROT_H
