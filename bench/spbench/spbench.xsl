<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform" >

<!-- Desire Nuentsa, Inria -->

<xsl:output method="html" indent="no"/>

<xsl:template match="/"> <!-- Root of the document -->
  <html>
    <head> 
      <style type="text/css"> 
        td { white-space: nowrap;}
      </style>
    </head>
    <body>
    <table border="1" width="100%" height="100%">
        <TR> <!-- Write the table header -->
        <TH>Matrix</TH> <TH>N</TH> <TH> NNZ</TH>  <TH> Sym</TH>  <TH> SPD</TH> <TH> </TH> 
          <xsl:for-each select="BENCH/AVAILSOLVER/SOLVER">
            <xsl:sort select="@ID" data-type="number"/>
            <TH> 
              <xsl:value-of select="TYPE" />
              <xsl:text></xsl:text>
              <xsl:value-of select="PACKAGE" />
              <xsl:text></xsl:text>
            </TH>
          </xsl:for-each>
        </TR>
        
        <xsl:for-each select="BENCH/LINEARSYSTEM">
          <TR> <!-- print statistics for one linear system-->
            <TH rowspan="4"> <xsl:value-of select="MATRIX/NAME" /> </TH>  
            <TD rowspan="4"> <xsl:value-of select="MATRIX/SIZE" /> </TD>
            <TD rowspan="4"> <xsl:value-of select="MATRIX/ENTRIES" /> </TD>
            <TD rowspan="4"> <xsl:value-of select="MATRIX/SYMMETRY" /> </TD>
            <TD rowspan="4"> <xsl:value-of select="MATRIX/POSDEF" /> </TD>
            <TH> Compute Time </TH> 
            <xsl:for-each select="SOLVER_STAT">
              <xsl:sort select="@ID" data-type="number"/>
              <TD> <xsl:value-of select="TIME/COMPUTE" /> </TD>
            </xsl:for-each>
          </TR>
          <TR> 
            <TH> Solve Time </TH> 
            <xsl:for-each select="SOLVER_STAT">
              <xsl:sort select="@ID" data-type="number"/>
              <TD> <xsl:value-of select="TIME/SOLVE" /> </TD>
            </xsl:for-each>
          </TR>
          <TR> 
            <TH> Total Time </TH> 
            <xsl:for-each select="SOLVER_STAT">
              <xsl:sort select="@ID" data-type="number"/>
              <xsl:choose>
                <xsl:when test="@ID=../BEST_SOLVER/@ID">
                  <TD style="background-color:red"> <xsl:value-of select="TIME/TOTAL" />  </TD>
                </xsl:when>
                <xsl:otherwise>
                  <TD>  <xsl:value-of select="TIME/TOTAL" /></TD>
                </xsl:otherwise>
              </xsl:choose>
            </xsl:for-each>
          </TR>
          <TR>
            <TH> Error </TH> 
            <xsl:for-each select="SOLVER_STAT">
              <xsl:sort select="@ID" data-type="number"/>
              <TD> <xsl:value-of select="ERROR" />
              <xsl:if test="ITER">
                <xsl:text>(</xsl:text> 
                <xsl:value-of select="ITER" />
                <xsl:text>)</xsl:text>
              </xsl:if> </TD>
            </xsl:for-each>
          </TR>
        </xsl:for-each>
    </table>
  </body>
  </html>
</xsl:template>

</xsl:stylesheet>