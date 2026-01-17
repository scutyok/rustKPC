Name "Headless_Route"

AbortOnDamage		FALSE	//	abort the script if we're damaged
AbortOnEnemySighted FALSE	//	Don't abort if we sight an enemy
TimeOut 999.0
FaceMoveDir			TRUE

Loop1:

StartAnim "Walk"

MoveTo CPathTarget0
MoveTo CPathTarget1
MoveTo CPathTarget2
MoveTo CPathTarget3

End