#include "mandelbrot.h"
#include<QtGui/QPainter>
#include<QtGui/QImage>
#include<QtGui/QMouseEvent>
#include<QtCore/QTime>

void MandelbrotWidget::resizeEvent(QResizeEvent *)
{
  if(size < width() * height())
  {
    std::cout << "reallocate buffer" << std::endl;
    size = width() * height();
    if(buffer) delete[]buffer;
    buffer = new unsigned char[4*size];
  }
}

void MandelbrotWidget::paintEvent(QPaintEvent *)
{
  QTime time;
  time.start();
  int alignedWidth = (width()/packetSize)*packetSize;
  real yradius = xradius * height() / width();
  vector2 start(center.x() - xradius, center.y() - yradius);
  vector2 step(2*xradius/width(), 2*yradius/height());
  int pix = 0, total_iter = 0;
  static float max_speed = 0;

  for(int y = 0; y < height(); y++)
  {
    // for each pixel, we're going to do the iteration z := z^2 + c where z and c are complex numbers, 
    // starting with z = c = complex coord of the pixel. pzi and pzr denote the real and imaginary parts of z.
    // pci and pcr denote the real and imaginary parts of c.

    packet pzi_start, pci_start;
    for(int i = 0; i < packetSize; i++) pzi_start[i] = pci_start[i] = start.y() + y * step.y();

    for(int x = 0; x < alignedWidth; x += packetSize, pix += packetSize)
    {
      packet pcr, pci = pci_start, pzr, pzi = pzi_start, pzr_buf;
      for(int i = 0; i < packetSize; i++) pzr[i] = pcr[i] = start.x() + (x+i) * step.x();

      // do the iterations. Every 4 iterations we check for divergence, in which case we can stop iterating.
      int j;
      for(j = 0; j < iter/4 && (pzr.cwiseAbs2() + pzi.cwiseAbs2()).eval().minCoeff() < 4; j++)
      {
        total_iter += 4 * packetSize;
        for(int i = 0; i < 4; i++)
        {
          pzr_buf = pzr;
          pzr = pzr.cwiseAbs2() - pzi.cwiseAbs2() + pcr;
          pzi = 2 * pzr_buf.cwiseProduct(pzi) + pci;
        }
      }

      // compute arbitrary pixel colors
      packet pblue, pgreen;
      if(j == iter/4)
      {
        packet pampl = (pzr.cwiseAbs2() + pzi.cwiseAbs2());
        pblue = real(510) * (packet::constant(0.1) + pampl).cwiseInverse().cwiseMin(packet::ones());
        pgreen = real(2550) * (packet::constant(10) + pampl).cwiseInverse().cwiseMin(packet::constant(0.1));
      }
      else pblue = pgreen = packet::zero();

      for(int i = 0; i < packetSize; i++)
      {
        buffer[4*(pix+i)] = (unsigned char)(pblue[i]);
        buffer[4*(pix+i)+1] = (unsigned char)(pgreen[i]);
        buffer[4*(pix+i)+2] = 0;
      }
    }

    // if the width is not a multiple of packetSize, fill the remainder in black
    for(int x = alignedWidth; x < width(); x++, pix++)
      buffer[4*pix] = buffer[4*pix+1] = buffer[4*pix+2] = 0;
  }
  int elapsed = time.elapsed();
  float speed = elapsed ? float(total_iter)*1000/elapsed : 0;
  max_speed = std::max(max_speed, speed);
  std::cout << elapsed << " ms elapsed, "
            << total_iter << " iters, "
            << speed << " iters/s (max " << max_speed << ")" << std::endl;

  QImage image(buffer, width(), height(), QImage::Format_RGB32);
  QPainter painter(this);
  painter.drawImage(QPointF(0,0), image);
}

void MandelbrotWidget::mousePressEvent(QMouseEvent *event)
{
  if( event->buttons() & Qt::LeftButton )
  {
    lastpos = event->pos();
    real yradius = xradius * height() / width();
    center = vector2(center.x() + (event->pos().x() - width()/2) * xradius * 2 / width(),
                     center.y() + (event->pos().y() - height()/2) * yradius * 2 / height());
    update();
  }
}

void MandelbrotWidget::mouseMoveEvent(QMouseEvent *event)
{
  QPoint delta = event->pos() - lastpos;
  lastpos = event->pos();
  if( event->buttons() & Qt::LeftButton )
  {
    real t = 1 + 3 * real(delta.y()) / height();
    if(t < 0.5) t = 0.5;
    if(t > 2) t = 2;
    xradius *= t;
    update();
  }
}

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);
  MandelbrotWidget w;
  w.show();
  return app.exec();
}

#include "mandelbrot.moc"
