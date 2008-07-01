#ifndef MANDELBROT_H
#define MANDELBROT_H

#include <Eigen/Array>
#include <QtGui/QApplication>
#include <QtGui/QWidget>
#include <QtCore/QThread>

class MandelbrotWidget;

class MandelbrotThread : public QThread
{
    friend class MandelbrotWidget;
    MandelbrotWidget *widget;
    long long total_iter;
    int id, max_iter;
    bool single_precision;

  public:
    MandelbrotThread(MandelbrotWidget *w, int i) : widget(w), id(i) {}
    void run();
    template<typename Real> void render(int img_width, int img_height);
};

class MandelbrotWidget : public QWidget
{
    Q_OBJECT

    friend class MandelbrotThread;
    Eigen::Vector2d center;
    double xradius;
    int size;
    unsigned char *buffer;
    QPoint lastpos;
    int draft;
    MandelbrotThread **threads;
    int threadcount;

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
      threadcount = QThread::idealThreadCount();
      threads = new MandelbrotThread*[threadcount];
      for(int th = 0; th < threadcount; th++) threads[th] = new MandelbrotThread(this, th);
    }
    ~MandelbrotWidget()
    {
      if(buffer) delete[]buffer;
      for(int th = 0; th < threadcount; th++) delete threads[th];
      delete[] threads;
    }
};

#endif // MANDELBROT_H
